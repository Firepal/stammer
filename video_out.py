from pathlib import Path
from frame_cache import LRUCacheBytes

import subprocess
from subprocess import Popen
import io
import logging
import re
import os

magic_headers = {
    "png": b'\x89PNG',
    "qoi": b'qoif',
}

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

def extract_frames_to_disk(frames_dir, carrier_path, used_frames, image_format):
    # Batch in chunks to ensure reasonable command length
    chunk_len = 80
    frame_chunks = chunks_of_n(used_frames, chunk_len)

    logging.info("extracting required frames to disk:")

    ext = image_format

    for chunk_index, frame_chunk in enumerate(frame_chunks):
        frame_strings = [str(frame) for frame in frame_chunk]
        select_string = "select='eq(n\\," + ")+eq(n\\,".join(frame_strings) + ")'"

        call = [
            'ffmpeg',
            '-v', 'quiet',
            '-i', str(carrier_path),
            '-vf', select_string,
            "-fps_mode", "passthrough",
            str(frames_dir / f'temp%06d.{ext}')
        ]

        print(f"Decoding chunk {chunk_index+1} of {len(frame_chunks)}", end='\r')

        subprocess.run(call,check=True)

        for i, frame_i in enumerate(frame_chunk):
            os.rename(
                frames_dir / (f'temp%06d.{ext}' % (i + 1,)),
                frames_dir / (f'frame%06d.{ext}' % (frame_i,))
            )

    print()

class VideoHandler:
    def __init__(self,
            carrier_path: Path,
            temp_dir: Path,
            carrier_frame_length: float,
            carrier_framecount: int,
            output_framecount: int,
            image_format: string
        ):
        self.output_frame_count = output_framecount

        self.carrier_path = carrier_path
        self.temp_dir = temp_dir
        self.frames_dir = self.temp_dir / 'frames'

        self.framecount = int(carrier_framecount)
        self.frame_length = carrier_frame_length

        self.image_format = image_format

        self.frames_written = 0
    
    def get_frame(self, carrier_idx: int, output_idx = None):
        assert(carrier_idx < self.framecount)

    # returns True if frame should be encoded
    def write_frame(self, idx: int):
        self.frames_written += 1
        self.print_progress()
        return True

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

class VideoHandlerDisk(VideoHandler):
    def __init__(self, *args):
        super().__init__(*args)
    
    def get_frame(self, carrier_idx, output_idx = None):
        super().get_frame(carrier_idx)
        
        return open(self.frames_dir / f"frame{idx:06d}.png", 'rb')

    def preprocess_frames(self, frames_map: dict, frames_used: list):
        extract_frames_to_disk(self.frames_dir, self.carrier_path, frames_used, self.image_format)


# assumes the next magic header is the end of the current substream
# so we don't need to scan for per-format footers
def substream_indices(stream: bytes, magic_header: bytes):
    headers = [mh.start() for mh in re.finditer(re.escape(magic_header), stream)]
    if len(headers) == 0:
        return []

    footers = headers[1:]
    footers.append(len(stream))

    return [(h, f) for h, f in zip(headers, footers)]

class VideoHandlerMem(VideoHandler):

    def __init__(self, *args):
        super().__init__(*args)
        self.cache = LRUCacheBytes()
        self.cache_hits = 0

        self.frame_jobs = {}
        self.frame_chunks = None

        self.frames_lookahead = 1

    def get_frame(self, carrier_idx: int, output_idx = None) -> io.BytesIO:
        super().get_frame(carrier_idx, output_idx)
        self.cache.process()

        frame = self._get_cached_frame(carrier_idx)

        if output_idx is not None:
            cache_ratio = self.cache.current_bytes / self.cache.max_bytes
            if cache_ratio < 0.95:
                self._prefetch(output_idx)

            if self.is_last_occurence(carrier_idx, output_idx):
                self.cache.pop_at(carrier_idx)

        return io.BytesIO(frame)

    def preprocess_frames(self, frames_map: dict, frames_used: list):
        # logging.info("making chunks")
        # self.frame_chunks = make_chunks_merging(frames_used, 10)
        self.frames_map = frames_map
        self.lomap = self.make_last_occurence_map(frames_map)

    def get_progress_strings(self):
        strs = super().get_progress_strings()
        strs.append(f"{self.cache_hits} cache hits")
        strs.append(f"{self.cache.current_bytes / (1024 * 1024):.2f} MiB / {len(self.cache.items)} cached frames")
        return strs

    def _prefetch(self, output_frame: int):
        i = 0
        while len(self.frame_jobs) <= 4:
            future_output_frame = output_frame + i
            if future_output_frame > self.output_frame_count-1:
                break
            i += 1

            future_carrier_frames = self.frames_map[future_output_frame]
            for carrier_frame in future_carrier_frames:
                if carrier_frame in self.frame_jobs:
                    # already being worked on
                    continue

                if not self.cache.item_usable(carrier_frame):
                    self.make_frame_job(carrier_frame)

    def make_last_occurence_map(self, frames_map: dict):
        logging.info("making last occurence map (cache optimization)")
        lomap = {}
        for output_frame in sorted(frames_map.keys()):
            carrier_frames = frames_map[output_frame]
            for carrier_frame in carrier_frames:
                lomap[carrier_frame] = output_frame

        return lomap

    def is_last_occurence(self, carrier_idx: int, output_idx: int):
        return self.lomap[carrier_idx] == output_idx

    def frame_call(self, start_frame: int, end_frame: int):
        start_time = start_frame * self.frame_length
        frame_count = end_frame - start_frame
        return [
            'ffmpeg',
            '-v', 'quiet',
            '-ss', str(start_time),
            '-i', str(self.carrier_path),
            '-c:v', self.image_format,
            '-frames:v', str(frame_count),
            '-f', 'image2pipe',
            '-'
        ]

    def _get_cache_span(self, carrier_frame):
        min_f = max(carrier_frame, 0)
        max_f = min(carrier_frame + 1, self.framecount)

        # chunk = self.get_frame_chunk_for_frame(carrier_frame)
        # if chunk != None:
        #     min_f, max_f = chunk
        return (min_f, max_f)

    def _frames_into_cache(self, decoded_frames: bytes, start_frame: int, end_frame: int):
        magic = magic_headers[self.image_format]
        indices = substream_indices(decoded_frames, magic)

        for i, carrier_frame in enumerate(range(start_frame, end_frame)):
            start, end = indices[i]

            frame_slice = decoded_frames[start:end]
            self.cache.set_item(carrier_frame, frame_slice)

    def _complete_frame_job(self, carrier_frame: int):
        frame_proc, (start_frame, end_frame) = self.frame_jobs[carrier_frame]
        frames, _ = frame_proc.communicate()

        self._frames_into_cache(frames, start_frame, end_frame)
        del self.frame_jobs[carrier_frame]

    def make_frame_job(self, carrier_idx: int):
        start_frame, end_frame = self._get_cache_span(carrier_idx)
        cmd = self.frame_call(start_frame, end_frame)
        frame_proc = Popen(cmd, stdout=subprocess.PIPE)

        frame_job = (frame_proc, (start_frame, end_frame))
        self.frame_jobs[carrier_idx] = frame_job
        return frame_job

    def _get_cached_frame(self, carrier_idx: int):
        if self.cache.item_usable(carrier_idx):
            self.cache_hits += 1
        else:
            try:
                frame_job = self.frame_jobs[carrier_idx]
            except KeyError:
                frame_job = self.make_frame_job(carrier_idx)

            self._complete_frame_job(carrier_idx)

        return self.cache.get_item(carrier_idx)

class VideoHandlerDummyCollector(VideoHandler):
    def __init__(self, *args):
        super().__init__(*args)

        # output frame -> carrier frames used to produce it
        self.output_to_carrier = {}

        self.frames_total = set()
        self._frames_used = set()

    def get_frame(self, carrier_idx: int, *args):
        self._frames_used.add(carrier_idx)

    def _collect(self, output_idx: int):
        if len(self._frames_used) > 0:
            self.frames_total.update(self._frames_used)

            self.output_to_carrier[output_idx] = self._frames_used
            self._frames_used = set()

    def write_frame(self, output_idx: int):
        self._collect(output_idx)
        return False
