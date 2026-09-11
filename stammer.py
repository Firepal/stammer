#!/usr/bin/env python3

from argparse import ArgumentParser

from scipy.io import wavfile
from pathlib import Path
import subprocess
import io
import re

import tempfile
import logging


from audio_matching import BasicAudioMatcher, CombinedFrameAudioMatcher, UniqueAudioMatcher, WeightedAudioMatcher

from video_out import VideoHandler, VideoHandlerDisk, VideoHandlerMem, VideoHandlerDummyCollector
from video_builder import VideoBuilderBasic, VideoBuilderCombined

TEMP_DIR = Path('temp')

DEFAULT_FRAME_LENGTH = 1/25 # Seconds

BAND_WIDTH = 1.2
INTERNAL_SAMPLERATE = 44100 # Hz

COMMON_AUDIO_EXTS = [
    "wav",
    "wv",
    "mp3",
    "m4a",
    "aac",
    "ogg",
    "opus",
]

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('carrier_path', type=Path, metavar='carrier_track', help='path to an audio or video file that frames will be taken from')
    parser.add_argument('modulator_path', type=Path, metavar='modulator_track', help='path to an audio or video file that will be reconstructed using the carrier track')
    parser.add_argument('output_path', type=Path, metavar='output_file', help='path to file that will be written to; should have an audio or video file extension (such as .wav, .mp3, .mp4, etc.)')
    parser.add_argument('-f', '--custom-frame-length', help='uses this number as frame length, in seconds. defaults to 0.04 seconds (1/25th of a second) for audio, or the real frame rate for video')
    parser.add_argument('-vm', '--video_mode', choices=('disk', 'ram'), default='ram', help='How STAMMER will store video frames internally.\
                        disk: Copy frames to temp directory (cleaned after the program closes).\
                        ram: Decode frames into system memory as needed, deleting least recently used frames over time. Recommended for very large videos.')
    parser.add_argument('-mb', '--cache_mibibytes', type=int, default=400, help='Only applies to "ram" video mode. The memory cap, in mibibytes, for the video frame cache.')
    parser.add_argument('-if', '--image_format', choices=('png','qoi'), default='qoi', help='Internal image format for frames.')
    parser.add_argument('-m', '--matcher_mode', choices=('basic', 'combination', 'unique', 'weighted'), default='basic', help="""Which algorithm Stammer will use.
        basic: replace each frame in the modulator with the most similar frame in the carrier.
        combination: replace each frame in the modulator with a linear combination of several frames in the carrier, to more closely approximate it.
        unique: limit each carrier frame to only appear once. If the carrier is longer than the modulator, some carrier frames will not be played, if it is shorter than the modulator, the modulator will be trimmed to the length of the carrier.
        weighted: apply an A-weighting curve to the audio spectra, to try and make formants more similar.""")

    return parser


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
    return float(subprocess.run(
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
        ).stdout)

def get_framecount(path):
    result = subprocess.run(
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
    # Prevent the error "ValueError: could not convert string to float: '56\n\n56\n'"
    return float(re.findall(r"\d+",result)[0])

def is_audio_filename(name):
    _path = Path(name)

    # Turns out Path.suffixes is empty for dotfiles (".mp4").
    ext = _path.suffixes[-1] if _path.suffixes else _path.stem

    return ext[1:] in COMMON_AUDIO_EXTS

def convert_audio_to_wav(in_path, out_path, channels=None):
    call = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-i', str(in_path),
        '-vn', '-map', '0:a:0',
        '-ar', str(INTERNAL_SAMPLERATE),
        '-c:a', 'pcm_s16le',
        '-f', 'wav',
        str(out_path)
    ]
    if channels is not None:
        idx = call.index('-ar')
        call[idx:idx] = ['-ac', str(channels)]
    subprocess.check_output(call)

def encode_audio(audio_path, output_path):
    subprocess.run(
        [
            'ffmpeg',
            '-loglevel', 'error',
            '-y', '-i', str(audio_path),
            str(output_path)
        ],
        check=True
    )

def collect_builder_frames(builder, video_handler_args):
    # this collects frame indices and does nothing else
    video_handler_dummy = VideoHandlerDummyCollector(*video_handler_args)

    old_video_handler = builder.video_handler

    builder.video_handler = video_handler_dummy
    builder.process()

    builder.video_handler = old_video_handler

    frames_used = video_handler_dummy.frames_total
    frames_map = video_handler_dummy.output_to_carrier
    del video_handler_dummy

    return (frames_used, frames_map)

def process(
    carrier_path, modulator_path, output_path,
    custom_frame_length, matcher_mode, video_mode,
    image_format, cache_mibibytes
    ):
    if not carrier_path.is_file():
        raise FileNotFoundError(f"Carrier file {carrier_path} not found.")
    if not modulator_path.is_file():
        raise FileNotFoundError(f"Modulator file {modulator_path} not found.")

    carrier_type = file_type(carrier_path)
    modulator_type = file_type(modulator_path)

    if not (('video' in modulator_type) or ('audio' in modulator_type)):
        logging.error(f"Unrecognized modulator file type: {modulator_path}. Should be audio or video")
        return

    if not (('video' in carrier_type) or ('audio' in carrier_type)):
        logging.error(f"Unrecognized file type: {carrier_path}. Should be audio or video")
        return

    carrier_duration = get_duration(carrier_path)
    modulator_duration = get_duration(modulator_path)

    carrier_is_video = 'video' in carrier_type
    output_is_video = carrier_is_video and not is_audio_filename(output_path)

    if output_is_video:
        logging.info("Calculating video length")

        carrier_framecount = get_framecount(carrier_path)
        carrier_frame_length = carrier_duration / carrier_framecount
        frame_length = carrier_frame_length

    frame_length = DEFAULT_FRAME_LENGTH

    if custom_frame_length is not None:
        frame_length = float(custom_frame_length)
    elif output_is_video:
        frame_length = carrier_frame_length

    frame_length = min(frame_length, carrier_duration / 3, modulator_duration / 3)


    logging.info("converting audio for processing")

    def make_audio(path_in, outname, channels=None):
        path_out = TEMP_DIR / outname
        convert_audio_to_wav(path_in, path_out, channels)

        return wavfile.read(path_out, mmap = True)[1]

    carrier_audio_mono = make_audio(carrier_path, "in_carrier_mono.wav", channels=1)
    modulator_audio_mono = make_audio(modulator_path, "in_modulator_mono.wav", channels=1)

    logging.info("analyzing audio")

    matcher_args = (carrier_audio_mono, modulator_audio_mono, INTERNAL_SAMPLERATE, frame_length)
    match matcher_mode:
        case "basic":
            audio_matcher = BasicAudioMatcher(*matcher_args)
        case "combination":
            audio_matcher = CombinedFrameAudioMatcher(*matcher_args)
        case "unique":
            audio_matcher = UniqueAudioMatcher(*matcher_args)
        case "weighted":
            audio_matcher = WeightedAudioMatcher(*matcher_args)

    audio_path = TEMP_DIR / 'out.wav'

    logging.info("creating output audio")
    carrier_audio_full = make_audio(carrier_path, "in_carrier_full.wav")
    audio_matcher.make_output_audio(carrier_audio_full, audio_path)

    # we won't need to recompute best_matches
    audio_matcher.free_match_data()

    if not output_is_video:
        encode_audio(audio_path, output_path)
        return

    # at this point it's guaranteed we're outputting a video

    output_framecount = int(len(audio_matcher.best_matches) * audio_matcher.frame_length / carrier_frame_length)

    video_handler_args = (
        carrier_path,
        TEMP_DIR,
        carrier_frame_length,
        carrier_framecount,
        output_framecount,
        image_format
    )

    if video_mode == "ram":
        video_handler = VideoHandlerMem(*video_handler_args)
        video_handler.cache.max_bytes = cache_mibibytes << 20
    elif video_mode == "disk":
        video_handler = VideoHandlerDisk(*video_handler_args)

    builder_args = (video_handler, audio_matcher, output_path)

    if type(audio_matcher) in (BasicAudioMatcher, UniqueAudioMatcher, WeightedAudioMatcher):
        builder = VideoBuilderBasic(*builder_args)
    elif type(audio_matcher) == CombinedFrameAudioMatcher:
        builder = VideoBuilderCombined(*builder_args)

    logging.info("precalculating required frames")
    frames_used, frames_map = collect_builder_frames(builder, video_handler_args)
    video_handler.preprocess_frames(frames_map, frames_used)

    logging.info("building output video")
    builder.process()
    builder.close_encoder()

def main():
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    # check required command line tools
    test_command(['ffmpeg', '-version'])
    test_command(['ffprobe', '-version'])

    args = get_parser().parse_args()
    with tempfile.TemporaryDirectory() as tempdir:
        global TEMP_DIR
        TEMP_DIR = Path(tempdir)
        process(**vars(args))

if __name__ == '__main__':
    main()
