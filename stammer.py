#!/usr/bin/env python3

from argparse import ArgumentParser
from typing import List
import numpy as np
from scipy.io import wavfile
from pathlib import Path
import shutil
import subprocess
import sys
import io

from PIL import Image
import tempfile
import logging

from src.audio_matching import AudioMatcher, BasicAudioMatcher, CombinedFrameAudioMatcher, UniqueAudioMatcher
from src.framegetter import FrameGetter, FrameGetterDisk, FrameGetterMem
from src.video_out import VideoOut


TEMP_DIR = Path('temp')

MAX_BASIS_WIDTH = 6
MAX_TESSELLATION_COUNT = 9
DEFAULT_FRAME_LENGTH = 1/25 # Seconds

BAND_WIDTH = 1.2
INTERNAL_SAMPLERATE = 44100 # Hz


# max number of frames stored in memory
MEM_DECAY_MAX = 500

COMMON_AUDIO_EXTS = [
    "wav",
    "wv",
    "mp3",
    "m4a",
    "aac",
    "ogg",
    "opus",
]

def test_command(cmd):
    try:
        subprocess.run(cmd, capture_output=True)
    except FileNotFoundError as error:
        logging.error(f"ERROR: '{cmd[0]}' not found. Please install it.")
        raise error

def file_type(path):
    # is the file at path an audio file, video file, or neither?
    return subprocess.run(
        [
            'ffprobe',
            '-loglevel', 'error',
            '-show_entries', 'stream=codec_type',
            '-of', 'csv=p=0',
            str(path)
        ],
        capture_output=True,
        check=True,
        text=True
    ).stdout

def get_duration(path):
    return subprocess.run(
            [
                'ffprobe',
                '-i', str(path),
                '-show_entries', 'format=duration',
                '-v', 'quiet',
                '-of', 'csv=p=0'
            ],
            capture_output=True,
            check=True,
            text=True
        ).stdout

def get_framecount(path):
    return subprocess.run(
            [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-count_packets',
                '-show_entries', 'stream=nb_read_packets',
                '-print_format', 'csv=p=0',
                str(path)
            ],
            capture_output=True,
            check=True,
            text=True
        ).stdout



def build_output_video(framegetter: FrameGetter, matcher: AudioMatcher):
    logging.info("building output video")

    video_out = VideoOut(framegetter.temp_dir,1.0/framegetter.frame_length,framegetter.output_path)
    
    for video_frame_i in range(framegetter.best_match_count):
        matcher.process_video_frame(video_frame_i,framegetter,video_out)
    
    # signals FrameGetterDisk to start encoding
    video_out.complete()

def is_audio_filename(name):
    return Path(name).suffixes[0][1:] in COMMON_AUDIO_EXTS

def get_audio_as_wav_bytes(path):
    ff_out = bytearray(subprocess.check_output(
        [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-i', str(path),
            '-vn', '-map', '0:a:0',
            '-ac', '1',
            '-ar', str(INTERNAL_SAMPLERATE),
            '-c:a', 'pcm_s16le',
            '-f', 'wav', '-'
        ]
    ))

    # fix file size in header length
    actual_data_len = len(ff_out)-44
    ff_out[4:8] = (actual_data_len).to_bytes(4,byteorder="little")

    return io.BytesIO(bytes(ff_out))

def process(carrier_path, modulator_path, output_path, custom_frame_length, matcher_mode, video_mode, color_mode, min_cached_frames):
    if not carrier_path.is_file():
        raise FileNotFoundError(f"Carrier file {carrier_path} not found.")
    if not modulator_path.is_file():
        raise FileNotFoundError(f"Modulator file {modulator_path} not found.")
    carrier_type = file_type(carrier_path)
    modulator_type = file_type(modulator_path)
    carrier_duration = float(get_duration(carrier_path))
    modulator_duration = float(get_duration(modulator_path))

    if 'video' in carrier_type:
        output_is_audio = is_audio_filename(output_path)
        carrier_is_video = not output_is_audio

        logging.info("Calculating video length")
        carrier_framecount = float(get_framecount(carrier_path))
        video_frame_length = carrier_duration / carrier_framecount
        
        frame_length = video_frame_length

    elif 'audio' in carrier_type:
        carrier_is_video = False
        frame_length = DEFAULT_FRAME_LENGTH
    else:
        logging.error(f"Unknown file type: {carrier_path}. Should be audio or video")
        return

    if not (('video' in modulator_type) or ('audio' in modulator_type)):
        logging.error(f"Unknown file type: {modulator_path}. Should be audio or video")
        return
    
    if not (custom_frame_length is None):
        frame_length = float(custom_frame_length)
    
    # what's this for?
    frame_length = min(frame_length, carrier_duration / 3)
    frame_length = min(frame_length, modulator_duration / 3)

    logging.info("reading audio")
    _, carrier_audio = wavfile.read(get_audio_as_wav_bytes(carrier_path))
    _, modulator_audio = wavfile.read(get_audio_as_wav_bytes(modulator_path))


    logging.info("analyzing audio")
    match matcher_mode:
        case "basic": matcher = BasicAudioMatcher(carrier_audio, modulator_audio, INTERNAL_SAMPLERATE, frame_length)
        case "combination": matcher = CombinedFrameAudioMatcher(carrier_audio, modulator_audio, INTERNAL_SAMPLERATE, frame_length)
        case "unique":  matcher = UniqueAudioMatcher(carrier_audio, modulator_audio, INTERNAL_SAMPLERATE, frame_length)
        case _: 
            logging.error("Unknown matcher mode")
            return

    logging.info("creating output audio")
    matcher.make_output_audio(TEMP_DIR / 'out.wav')

    if carrier_is_video:
        match video_mode:
            case "mem_decay":
                handler = FrameGetterMem(carrier_path,output_path,TEMP_DIR,matcher,carrier_framecount,video_frame_length,color_mode)
                handler.cache.decay = MEM_DECAY_MAX
                handler.set_min_cached_frames(min_cached_frames)
            case "disk":
                handler = FrameGetterDisk(carrier_path,output_path,TEMP_DIR,matcher,carrier_framecount,video_frame_length,color_mode)
            case _: 
                logging.error("Unknown video mode")
                return
        
        build_output_video(handler, matcher)
    else:
        subprocess.run(
            [
                'ffmpeg',
                '-loglevel', 'error',
                '-y', '-i', str(TEMP_DIR / 'out.wav'),
                str(output_path)
            ],
            check=True
        )

def main():
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    # check required command line tools
    test_command(['ffmpeg', '-version'])
    test_command(['ffprobe', '-version'])
    
    parser = ArgumentParser()
    parser.add_argument('carrier_path', type=Path, metavar='carrier_track', help='path to an audio or video file that frames will be taken from')
    parser.add_argument('modulator_path', type=Path, metavar='modulator_track', help='path to an audio or video file that will be reconstructed using the carrier track')
    parser.add_argument('output_path', type=Path, metavar='output_file', help='path to file that will be written to; should have an audio or video file extension (such as .wav, .mp3, .mp4, etc.)')
    parser.add_argument('--custom-frame-length', '-f', help='uses this number as frame length, in seconds. defaults to 0.04 seconds (1/25th of a second) for audio, or the real frame rate for video')
    parser.add_argument('-vm', '--video_mode', choices=('disk', 'mem_decay'), default='disk', help='How STAMMER will store video frames internally.\
                        disk: Copy all frames to temp directory.\
                        mem_decay: Decode frames into memory as needed and deletes unused frames over time. Recommended for very large videos.')
    parser.add_argument('-mcf', '--min_cached_frames', type=int, default=2, help='Only applies to "mem_decay" video mode. Minimum number of frames STAMMER will cache for one decayed frame.')
    parser.add_argument('-c', '--color_mode', choices=('8fast', '8full', 'full'), default='full', help='Bitdepth of internal video frames.\
                        8fast: generates 8-bit PNGs with default palette, fast and low filesize but low-quality. \
                        8full: generates 8-bit PNGs with a custom 256-color palette for each frame. slow but looks great. \
                        full: generates 16-bit PNGs, default. fast and looks good, but high filesize.')
    parser.add_argument('-m', '--matcher_mode', choices=('basic', 'combination', 'unique'), default='basic', help="""Which algorithm Stammer will use.
        basic: replace each frame in the modulator with the most similar frame in the carrier.
        combination: replace each frame in the modulator with a linear combination of several frames in the carrier, to more closely approximate it.
        unique: limit each carrier frame to only appear once. If the carrier is longer than the modulator, some carrier frames will not be played, if it is shorter than the modulator, the modulator will be trimmed to the length of the carrier.""")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tempdir:
        global TEMP_DIR
        TEMP_DIR = Path(tempdir)
        process(**vars(args))


if __name__ == '__main__':
    main()
