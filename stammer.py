from argparse import ArgumentParser
import numpy as np
from scipy.io import wavfile
from pathlib import Path
import shutil
import subprocess

TEMP_DIR = Path('temp')

DEFAULT_FRAME_LENGTH = 1/25 # Seconds

BAND_WIDTH = 1.2
INTERNAL_SAMPLERATE = 44100 # Hz

def test_command(cmd):
    try:
        subprocess.run(cmd, capture_output=True)
    except FileNotFoundError as error:
        print(f"ERROR: '{cmd[0]}' not found. Please install it.")
        raise error

def make_normalized_bands(frames_input,band_width):
    transforms = np.fft.fft(frames_input)
    spectra = abs(transforms[:,1:len(transforms[0])//2])
    split_points = [0]
    i = 2
    while i < len(spectra[0]):
        if int(i) > split_points[-1]:
            split_points.append(int(i))
        i *= band_width
    section_lengths = []
    for i in range(len(split_points) - 1):
        section_lengths.append(split_points[i+1]-split_points[i])
    section_lengths.append(len(spectra[0]) - split_points[-1])
    bands = np.divide(np.add.reduceat(spectra, split_points, axis=1), section_lengths) # average value in each vand
    vector_magnitudes = np.sqrt((bands * bands).sum(axis=1))
    vector_magnitudes[vector_magnitudes==0]=1
    normalized_bands = bands / vector_magnitudes[:,None]
    
    return normalized_bands

def make_frames(input_audio, frame_length):
    if input_audio.dtype != float:
        intmax = np.iinfo(input_audio.dtype).max
        input_audio = input_audio.astype(float) / intmax

    # todo: this will get slightly off over time.
    num_frames = (len(input_audio) // frame_length) - 2
    
    window = np.hanning(frame_length * 2)
    frames = np.zeros((num_frames, frame_length * 2), dtype=input_audio.dtype)

    for i in range(num_frames):
        a = input_audio[i*frame_length:i*frame_length+frame_length*2]
        frames[i] = (window * input_audio[i*frame_length:i*frame_length+frame_length*2])
        
    return frames

def find_best_match(carrier_bands, modulator_band):
    dot_products = np.sum(carrier_bands * modulator_band, axis=1)
    return np.argmax(dot_products)

def file_type(path):
    # is the file at path an audio file, video file, or neither?
    return subprocess.run(
        [
            'ffprobe',
            '-loglevel', 'error',
            '-show_entries', 'stream=codec_type',
            '-of', 'csv=p=0',
            path
        ],
        capture_output=True,
        check=True,
        text=True
    ).stdout

def get_duration(path):
    return subprocess.run(
            [
                'ffprobe',
                '-i', path,
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
                '-count_frames',
                '-show_entries', 'stream=nb_read_frames',
                '-print_format', 'csv=p=0',
                path
            ],
            capture_output=True,
            check=True,
            text=True
        ).stdout

def build_output_video(frames_dir, outframes_dir, best_matches, framerate, output_path):
    print("building output video")
        
    for i, match_num in enumerate(best_matches):
        shutil.copy(frames_dir / f'{match_num+1:08d}.png', outframes_dir / f'{i:08d}.png')
    subprocess.run(
        [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-y',
            '-framerate', str(framerate),
            '-i', outframes_dir / '%08d.jpeg',
            '-i', TEMP_DIR / 'out.wav',
            '-c:a', 'aac',
            '-shortest',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            output_path
        ],
        check=True
    )

def create_output_audio(best_matches, modulator_audio, carrier_frames, modulator_frames, samples_per_frame):
    output_audio = np.zeros(modulator_audio.shape, dtype=float)

    for i in range(len(modulator_frames)):
        carrier_frame = carrier_frames[best_matches[i]]
        modulator_frame = modulator_frames[i]
        modulator_frame_amp = np.sqrt(np.sum(modulator_frame*modulator_frame))
        carrier_frame_amp = np.sqrt(np.sum(carrier_frame*carrier_frame))
        if (carrier_frame_amp == 0):
            continue
        rescaled_frame = carrier_frame * (modulator_frame_amp / carrier_frame_amp)

        if (max(abs(rescaled_frame))) > 1:
            rescaled_frame /= max(abs(rescaled_frame))
        output_audio[i*samples_per_frame : i*samples_per_frame + samples_per_frame*2] += rescaled_frame

    wavfile.write(TEMP_DIR / 'out.wav', INTERNAL_SAMPLERATE, output_audio)

def separate_video_frames_disk(path):
    frames_dir = TEMP_DIR / 'frames'
    frames_dir.mkdir()
    subprocess.run(
        [
            'ffmpeg',
            '-loglevel', 'error',
            '-i', path,
            frames_dir / '%08d.jpg'
        ],
        check=True
    )
    return frames_dir

def separate_video_frames_mem(path):
    x = subprocess.check_output(
        [
            'ffmpeg',
            '-loglevel', 'error',
            '-i', path, '-c:v', 'png', 
            '-t', '4',
            '-f', 'image2pipe',
            '-'
        ]
    )
    return x

PNG_MAGIC = int("89504e47",16).to_bytes(4,byteorder='big')
JPG_MAGIC = int("ffd8ffe0",16).to_bytes(4,byteorder='big')
def split_frames(frames: bytes):
    print(frames.index(PNG_MAGIC))
    indices = []
    cur = 0
    while True:
        idx = frames.find(PNG_MAGIC, cur)
        if idx == -1: 
            indices.append(len(frames))
            break
        if not idx in indices: indices.append(idx)
        cur = max(cur+1,idx)
    
    new_frames = []
    for i in range(len(indices)-1):
        f_start = indices[i]
        f_end = indices[i+1]

        f = frames[f_start:f_end]
        new_frames.append(f)
    print(indices)
    return new_frames

def get_frame_ofs_index(frames: bytes, index):
    cur = 0
    total_idx = 0
    idx = 0
    while True:
        th = frames.find(PNG_MAGIC, cur)
        if th == -1: break
        idx = th
        if idx > cur: total_idx += 1
        if total_idx == index:
            break
        cur = max(cur,idx)

    return idx

def write_frames(frames: list[bytes]):
    for i, n in enumerate(frames):
        f = open(str(i)+".png","wb")
        f.write(n)
        f.close()


def process(carrier_path, modulator_path, output_path):
    if not carrier_path.is_file():
        raise FileNotFoundError(f"Carrier file {carrier_path} not found.")
    if not modulator_path.is_file():
        raise FileNotFoundError(f"Modulator file {modulator_path} not found.")


    carrier_type = file_type(carrier_path)
    modulator_type = file_type(modulator_path)

    if 'video' in carrier_type:
        carrier_is_video = True
        # carrier_duration = float(get_duration(carrier_path))
        # carrier_framecount = float(get_framecount(carrier_path))
        # frame_length = carrier_duration / carrier_framecount      
        print("Extracting video frames (this may take a while)")
        frames = separate_video_frames_mem(carrier_path)
        
        print("Splitting frames f")
        new_frames = split_frames(frames)
        print(len(new_frames))
        print(get_frame_ofs_index(frames,100))
        print("done")
    elif 'audio' in carrier_type:
        carrier_is_video = False
        frame_length = DEFAULT_FRAME_LENGTH
    else:
        print(f"Unrecognized file type: {carrier_path}. Should be audio or video")
        return

    if not (('video' in modulator_type) or ('audio' in modulator_type)):
        print(f"Unrecognized file type: {modulator_path}. Should be audio or video")
        return

    print("copying audio")
    subprocess.run(
        [
            'ffmpeg',
            '-loglevel', 'error',
            '-i', carrier_path,
            '-ac', '1',
            '-ar', str(INTERNAL_SAMPLERATE),
            TEMP_DIR / 'carrier.wav'
        ],
        check=True
    )
    subprocess.run(
        [
            'ffmpeg',
            '-loglevel', 'error',
            '-i', modulator_path,
            '-ac', '1',
            '-ar', str(INTERNAL_SAMPLERATE),
            TEMP_DIR / 'modulator.wav'
        ],
        check=True
    )

    print("reading audio")
    _, carrier_audio = wavfile.read(TEMP_DIR / 'carrier.wav')
    _, modulator_audio = wavfile.read(TEMP_DIR / 'modulator.wav')

    print("analyzing audio")
    samples_per_frame = int(frame_length * INTERNAL_SAMPLERATE)
    carrier_frames = make_frames(carrier_audio, samples_per_frame)
    modulator_frames = make_frames(modulator_audio, samples_per_frame)

    carrier_bands = make_normalized_bands(carrier_frames, BAND_WIDTH)
    modulator_bands = make_normalized_bands(modulator_frames, BAND_WIDTH)

    print("finding best matches")
    best_matches = []
    for i in range(len(modulator_bands)):
        best_matches.append(find_best_match(carrier_bands,modulator_bands[i]))


    print("creating output audio")
    create_output_audio(best_matches, modulator_audio, carrier_frames, modulator_frames, samples_per_frame)
    

    if carrier_is_video:
        outframes_dir = TEMP_DIR / 'outframes'
        outframes_dir.mkdir()
        build_output_video(frames, outframes_dir, best_matches, 1/frame_length, output_path)
    else:
        subprocess.run(
            [
                'ffmpeg',
                '-loglevel', 'error',
                '-y', '-i', TEMP_DIR / 'out.wav',
                output_path
            ],
            check=True
        )

def main():
    # check required command line tools
    test_command(['ffmpeg', '-version'])
    test_command(['ffprobe', '-version'])
    
    parser = ArgumentParser()
    parser.add_argument('carrier_path', type=Path, metavar='carrier_track', help='path to an audio or video file that frames will be taken from')
    parser.add_argument('modulator_path', type=Path, metavar='modulator_track', help='path to an audio or video file that will be reconstructed using the carrier track')
    parser.add_argument('output_path', type=Path, metavar='output_file', help='path to file that will be written to; should have an audio or video file extension (such as .wav, .mp3, .mp4, etc.)')
    args = parser.parse_args()
    
    try:
        TEMP_DIR.mkdir()
        process(**vars(args))
        shutil.rmtree(TEMP_DIR)
    except Exception:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)  # no guarantee that temp/ was created
        raise

if __name__ == '__main__':
    main()
