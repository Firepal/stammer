from pathlib import Path
from frame_cache import LRUCache
from audio_matching import AudioMatcher

import subprocess
import io
import logging
import re
import os

def apply_color_mode(ffmpeg_call,color_mode):
    color_strs = []
    if color_mode == '8fast':
        # color_strs = ['-pix_fmt', 'pal8', '-sws_dither', 'ed']
        color_strs = ['-pix_fmt', 'pal8']
    elif color_mode == '8full':
        color_strs = ['-vf', 'split[s0][s1];[s0]palettegen=256:0:stats_mode=single[p];[s1][p]paletteuse=new=1:dither=bayer']

    idx = ffmpeg_call.index('include_color_mode')
    ffmpeg_call.pop(idx)
    if color_mode != "full":
        for i, s in enumerate(color_strs):
            ffmpeg_call.insert(idx+i,s)
    return ffmpeg_call

def make_chunks_merging(frames: list[int], counting_lenience: int = 10):
    sorted_unique_frames = sorted(set(frames))

    start = end = sorted_unique_frames[0]
    chunks = []
    for n in sorted_unique_frames[1:]:
        if n - end <= counting_lenience:
            end = n
        else:
            if start == end:
                chunks.append((start, start+1))
            else:
                chunks.append((start, end))
            start = end = n
    chunks.append((start, end))

    return chunks

def chunks_of_n(input_list: list[int], n: int):
    chunks = []
    for i in range(0, len(input_list), n):
        chunks.append(input_list[i:n+i])
    return chunks

def extract_frames_to_disk(frames_dir, carrier_path, used_frames, color_mode):
    # Batch in chunks to ensure reasonable command length
    chunk_len = 80
    frame_chunks = chunks_of_n(used_frames, chunk_len)

    logging.info("extracting required frames to disk:")

    for chunk_index, frame_chunk in enumerate(frame_chunks):
        frame_strings = [str(frame) for frame in frame_chunk]
        select_string = "select='eq(n\\," + ")+eq(n\\,".join(frame_strings) + ")'"

        call = video_out.apply_color_mode([
            'ffmpeg',
            '-v', 'quiet',
            '-i', str(carrier_path),
            'include_color_mode',
            '-vf', select_string,
            "-fps_mode", "passthrough",
            str(frames_dir / 'temp%06d.png')
        ],color_mode)

        print(f"Decoding chunk {chunk_index+1} of {len(frame_chunks)}", end='\r')

        subprocess.run(call,check=True)

        for i, frame_i in enumerate(frame_chunk):
            os.rename(
                frames_dir / ('temp%06d.png' % (i + 1,)),
                frames_dir / ('frame%06d.png' % (frame_i,))
            )

    print()

class VideoHandler:
    def __init__(self, carrier_path: Path, output_path: Path, temp_dir: Path, matcher: AudioMatcher, framecount: int, frame_length: float, color_mode):
        self.matcher = matcher
        self.output_frame_count = int(len(matcher.get_best_matches()) * matcher.frame_length / frame_length)

        self.carrier_path = carrier_path
        self.output_path = output_path
        self.temp_dir = temp_dir
        self.frames_dir = self.temp_dir / 'frames'

        self.framecount = int(framecount)
        self.frame_length = frame_length

        self.color_mode = color_mode

        self.frames_written = 0
    
    def get_frame(self,idx):
        assert(idx < self.framecount)
    
    def write_frame(self, idx, frame: io.BytesIO):
        if not hasattr(self, 'out_proc'):
            self.out_proc = self.create_output_proc()

        frame.seek(0)
        self.out_proc.stdin.write(frame.read())

        self.frames_written += 1
        self.print_progress()

    def complete(self):
        print(end="\n")

        self.out_proc.communicate()

    def preprocess_frames(self, frames_map: dict, frames_used: list):
        pass

    # --- internal methods below

    def get_frame_chunk_for_frame(self, frame_idx: int):
        if self.frame_chunks is None:
            return None

        for i, chunk in enumerate(self.frame_chunks):
            if frame_idx < chunk[1] and frame_idx >= chunk[0]:
                return chunk

        return None

    def get_progress_strings(self) -> list[str]:
        strings: list[str] = []
        strings.append(str(self.frames_written) + "/" + str(self.output_frame_count))
        
        return strings
    
    def progress_strings_separated(self):
        ps = self.get_progress_strings()
        if len(ps) == 1: return ps[0]
        return " . ".join(self.get_progress_strings())
    
    def print_progress(self):
        print(self.progress_strings_separated(),end='      \r')

    def get_output_cmd(self):
        cmd = [
            'ffmpeg',
            '-v', 'quiet',
            '-y',
            '-framerate', str(1.0 / self.frame_length),
            '-f', 'image2pipe', '-i', 'pipe:',
            '-i', str(self.temp_dir / 'out.wav'),
            '-c:a', 'aac',
            '-c:v', 'libx264',
            '-crf', '24',
            '-pix_fmt', 'yuv420p',
            '-shortest',
            str(self.output_path)
        ]

        return cmd
    
    def create_output_proc(self):
        call = self.get_output_cmd()

        return subprocess.Popen(
            call,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL
        )

class VideoHandlerDisk(VideoHandler):
    def __init__(self, *args):
        super().__init__(*args)
    
    def get_frame(self,idx):
        super().get_frame(idx)
        
        return open(self.frames_dir / f"frame{idx:06d}.png", 'rb')

    def preprocess_frames(self, frames_map: dict, frames_used: list):
        extract_frames_to_disk(self.frames_dir, self.carrier_path, frames_used, self.color_mode)

PNG_HEADER = b"\x89PNG\r\n\x1a\n"
PNG_FOOTER = b"IEND\xaeB\x60\x82"

JPG_MAGIC = int("ffd8ffe0",16).to_bytes(4,byteorder='big')

def substream_indices(stream: bytes, magic_header: bytes, magic_footer: bytes):
    return [
        (mh.start(), mf.end())
        for mh, mf in zip(
            re.finditer(re.escape(magic_header), stream),
            re.finditer(re.escape(magic_footer), stream)
        )
    ]

class VideoHandlerMem(VideoHandler):
    def __init__(self, *args):
        super().__init__(*args)
        self.cache = LRUCache()
        self.cache_hits = 0

        self.frames_lookahead = 2

        self.frame_chunks = None
    
    def get_frame(self, idx) -> io.BytesIO:
        super().get_frame(idx)
        self.cache.process()
        
        if self.cache.item_usable(idx):
            self.cache_hits += 1
        else:
            self.__cache_frames(idx)
        
        frame = self.cache.get_item(idx)
        return io.BytesIO(frame)

    def preprocess_frames(self, frames_map: dict, frames_used: list):
        logging.info("making chunks")
        self.frame_chunks = make_chunks_merging(frames_used, 10)

    def get_progress_strings(self):
        strs = super().get_progress_strings()
        strs.append(f"{self.cache_hits} cache hits")
        strs.append(f"{self.cache.current_bytes / (1024 * 1024):.2f} MiB / {len(self.cache.items)} cached frames")
        return strs

    def set_min_cached_frames(self,mcf):
        self.frames_lookahead = mcf
        
    def __get_video_frames_mem(self, start_frame: int, end_frame: int):
        start_time = start_frame * self.frame_length
        call = apply_color_mode([
                'ffmpeg',
                '-loglevel', 'error',
                '-ss', str(start_time),
                '-i', self.carrier_path,
                '-c:v', 'png',
                'include_color_mode',
                '-frames:v', str(end_frame - start_frame),
                '-f', 'image2pipe',
                '-'
            ],self.color_mode)
        
        return subprocess.check_output(call)
    
    def __cache_frames(self,match_id):
        min_f = max(match_id, 0)
        max_f = min(match_id + self.frames_lookahead, self.framecount)

        chunk = self.get_frame_chunk_for_frame(match_id)
        if chunk != None:
            min_f, max_f = chunk

        decoded_frames = self.__get_video_frames_mem(min_f,max_f)
        new_frame_idxs = range(min_f, max_f)
        
        indices = substream_indices(decoded_frames, PNG_HEADER, PNG_FOOTER)

        for i, idx in enumerate(new_frame_idxs):
            start, end = indices[i]

            frame_slice = decoded_frames[start:end]
            self.cache.set_item(idx, frame_slice)

class VideoHandlerDummyCollector(VideoHandler):
    def __init__(self, *args):
        super().__init__(*args)

        # output frame -> carrier frames used to produce it
        self.output_to_carrier = {}

        self.frames_total = set()
        self._frames_used = set()

    def get_frame(self, idx: int):
        self._frames_used.add(idx)

    def write_frame(self, idx: int, *args):
        if len(self._frames_used) > 0:
            self.frames_total.update(self._frames_used)

            self.output_to_carrier[idx] = self._frames_used
            self._frames_used = set()

    def complete(self): pass
