from pathlib import Path
from src.decay_cache import DecayCache
import subprocess
import io

def apply_color_mode(ffmpeg_call,color_mode):
    color_strs = []
    if color_mode == '8fast':
        # color_strs = ['-pix_fmt', 'pal8', '-sws_dither', 'ed']
        color_strs = ['-pix_fmt', 'pal8']
    elif color_mode == '8full':
        color_strs = ['-vf', 'split[s0][s1];[s0]palettegen=256:0:stats_mode=single[p];[s1][p]paletteuse=new=1:dither=bayer']

    idx = ffmpeg_call.index('include_color_mode')
    if idx == -1:
        return ffmpeg_call
    ffmpeg_call.pop(idx)
    if color_mode != "full":
        for i, s in enumerate(color_strs):
            ffmpeg_call.insert(idx+i,s)
    return ffmpeg_call

class FrameGetter:
    def __init__(self, carrier_path: Path, output_path: Path, temp_dir: Path, matcher, framecount: int, frame_length: float, color_mode):
        self.matcher = matcher
        self.best_match_count = int(len(matcher.get_best_matches()) * matcher.frame_length / frame_length)

        self.carrier_path = carrier_path
        self.output_path = output_path
        self.temp_dir = temp_dir
        self.frames_dir = self.temp_dir / 'frames'

        self.framecount = int(framecount)
        self.frame_length = frame_length

        self.color_mode = color_mode
    
    def get_frame(self,idx):
        try:
            assert(idx < self.framecount)
        except AssertionError:
            print("ERROR:")
            print(f"STAMMER just tried to use carrier frame {idx}")
            print(f"but carrier only has {self.framecount} frames.")
            print()
            print("This should not happen.")
            print("\nQuitting.")
            quit()

    def complete(self):
        print(end="\n")
    
    def get_input_command(self, pre = [], post = []):
        call = [
                'ffmpeg',
                '-loglevel', 'error',
                'pre',
                '-i', self.carrier_path,
                '-c:v', 'mjpeg',
                'post'
        ]
        
        pairs = [('pre',pre),('post',post)]
        for pair_id, pair_val in pairs:
            idx = call.index(pair_id)
            if idx == -1: continue
            call.pop(idx)
            for i, v in enumerate(pair_val):
                call.insert(idx+i,v)
        return call

# TODO: 
# - Maybe have a decaying disk cache so we don't have to write all frames
class FrameGetterDisk(FrameGetter):
    def __init__(self, *args):
        super().__init__(*args)
        self.separate_frames()

    def separate_frames(self):
        print("Separating video frames")
        frames_dir = self.temp_dir / 'frames'
        frames_dir.mkdir(exist_ok=True)

        call = self.get_input_command(pre=['-stats'],post=['-b:v', '4M', str(frames_dir / 'frame%06d.jpg')])
        
        subprocess.run(call,check=True)
    
    def get_frame(self,idx):
        super().get_frame(idx)
        
        # Video frame filenames start at 1, not 0
        idx += 1
        return open(self.frames_dir / f"frame{idx:06d}.jpg", 'rb')

    def complete(self):
        pass


PNG_MAGIC = int("89504e47",16).to_bytes(4,byteorder='big')
JPG_MAGIC = int("ffd8ffe0",16).to_bytes(4,byteorder='big')
# JXL_MAGIC = int("0000000C4A584C200D0A870A",16).to_bytes(4,byteorder='big')

class FrameGetterMem(FrameGetter):
    def __init__(self, *args):
        super().__init__(*args)
        self.cache = DecayCache(self.framecount)
        self.cache_hits = 0

        self.frame_length_max = self.frame_length / max(self.frame_length,self.matcher.frame_length)
        self.frames_backtrack = 0
        self.frames_lookahead = int(max(1.0/self.frame_length_max,2))

    def set_min_cached_frames(self,mcf):
        # if a decayed frame is about to be used, we fetch the frame + this amount of frames around it
        # it's likely that the modulator will generally fetch similar frames
        self.frames_backtrack = 0
        self.frames_lookahead = int(max(1.0/self.frame_length_max,mcf))
        
        # This enforces that cached frame count cannot exceed decay time
        # i.e. if decay time is 500 frames, max cached frames will be 500
        self.cache.decay /= self.frames_lookahead

    def __get_video_frames_mem(self,start_frame: int,end_frame: int):
        start_time = start_frame * self.frame_length
        call = self.get_input_command(
            pre=['-ss', str(start_time)],
            post=['-frames:v', str(end_frame-start_frame),
            '-f', 'image2pipe','-']
        )
        
        return subprocess.check_output(call)
    
    def __get_frame_ofs_index(frames: bytes, index):
        cur = 0
        total_idx = 0
        idx = 0
        while True:
            check = frames.find(PNG_MAGIC, cur)
            if check == -1: break
            idx = check
            if idx > cur: total_idx += 1
            if total_idx == index: break
            cur = max(cur,idx+1)

        return idx

    def __get_frame_slice(frames: bytes, index: int):
        start = FrameGetterMem.__get_frame_ofs_index(frames,index)
        end = FrameGetterMem.__get_frame_ofs_index(frames,index+1)
        if start == end: end = len(frames)

        return frames[start:end]

    def __cache_decayed_frames(self,match_id):
        def grow_to_nondecayed(min,max):
            for idx in range(min,max):
                if self.cache.item_usable(idx): return idx
            return max

        min_f = max(match_id-self.frames_backtrack,0)
        max_f = min(match_id+self.frames_lookahead,self.framecount)

        min_f = grow_to_nondecayed(min_f,match_id)
        max_f = grow_to_nondecayed(match_id,max_f)

        decoded_frames = self.__get_video_frames_mem(min_f,max_f)
        
        new_frame_ids = range(min_f, max_f)
        self.cache.clear(new_frame_ids)
        for idx in new_frame_ids:
            frame_slice = FrameGetterMem.__get_frame_slice(decoded_frames,idx-min_f)
            self.cache.set_item(idx,frame_slice)
    
    def get_frame(self,idx) -> io.BytesIO:
        super().get_frame(idx)
        self.cache.process()
        
        if self.cache.item_usable(idx):
            self.cache_hits += 1
            self.cache.clear([idx])
        else:
            self.__cache_decayed_frames(idx)
        
        frame = self.cache.items[idx].item
        return io.BytesIO(frame)

    # def get_progress_strings(self):
    #     strs = super().get_progress_strings()
    #     strs.append(f"{self.cache_hits} cache hits")
    #     strs.append(f"{self.framecount-self.cache.decayed_items}/{self.framecount} cached frames")
    #     return strs
    
    
    def complete(self):
        pass