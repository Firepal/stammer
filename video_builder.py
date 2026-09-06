import math
import io
from typing import List
import numpy as np

from PIL import Image

import image_tiling
import fraction_bits

from video_out import VideoHandler, VideoHandlerDummyCollector
from audio_matching import AudioMatcher

MAX_BASIS_WIDTH = 6
MAX_TESSELLATION_COUNT = 9

class VideoBuilder:
    def __init__(self, video_handler: VideoHandler, audio_matcher: AudioMatcher):
        self.video_handler = video_handler
        self.audio_matcher = audio_matcher

class VideoBuilderBasic(VideoBuilder):
    def basic_video_frame(
            self,
            video_frame_i,
            best_matches,
            video_frame_length,
            audio_frame_length
            ):
        elapsed_time = video_frame_i * video_frame_length

        audio_frame_i = int(elapsed_time / audio_frame_length)

        # fractional time past start of audio frame
        audio_frame_fract = elapsed_time - (audio_frame_i * audio_frame_length)

        match_num = best_matches[audio_frame_i]
        elapsed_time_in_carrier = match_num * audio_frame_length + audio_frame_fract

        carrier_video_frame = int(math.floor(elapsed_time_in_carrier / video_frame_length))
        # print(match_num, " vs ", carrier_video_frame)
        
        return carrier_video_frame
    
    def process(self):
        best_matches = self.audio_matcher.get_best_matches()

        video_frame_length = self.video_handler.frame_length
        audio_frame_length = self.audio_matcher.frame_length

        self.output_frame_count = int(len(self.audio_matcher.best_matches) * audio_frame_length / video_frame_length)

        for video_frame_i in range(self.output_frame_count):
            carrier_video_index = self.basic_video_frame(
                video_frame_i, 
                best_matches,
                self.video_handler.frame_length, 
                self.audio_matcher.frame_length
            )
            carrier_video_index = min(carrier_video_index, int(self.video_handler.framecount - 1))

            carrier_vframe_bytes = self.video_handler.get_frame(carrier_video_index)
            self.video_handler.write_frame(video_frame_i, carrier_vframe_bytes)

class VideoBuilderCombined(VideoBuilder):
    def tesselate_composite(self, match_row, basis_coefficients, i):
        tiles: List[Image.Image] = []
        bits: List[List[int]] = []
        used_coeffs = [(j, coefficient) for j, coefficient in enumerate(basis_coefficients) if coefficient != 0]
        
        for k, coeff in used_coeffs:
            frame_num = min(match_row[k], self.video_handler.framecount - 1)
            frame_bytes = self.video_handler.get_frame(frame_num)

            if type(self.video_handler) is VideoHandlerDummyCollector:
                continue

            tiles.append(Image.open(frame_bytes))
            hot_bits,_ = fraction_bits.as_array(coeff)
            bits.append(hot_bits)

        img_bytes = io.BytesIO()

        if len(tiles) == 0 \
        or type(self.video_handler) is VideoHandlerDummyCollector:
            return img_bytes

        tesselation = image_tiling.Tiling(height=tiles[0].height,width=tiles[0].width)
        output_frame = Image.new('RGB',(tiles[0].width, tiles[0].height))
        
        for m in np.arange(1,MAX_TESSELLATION_COUNT):
            first_hot = next(((offset, x) for offset, x in enumerate(bits) if x[m]), None)
            if first_hot is not None:
                do_tile = tesselation.needs_tiling
                tb = tiles[first_hot[0]].copy()
                x0, y0, w, h = tesselation.get_image_placement()
                tb.thumbnail((w,h))
                output_frame.paste(tb, (x0,y0))
                if do_tile:
                    output_frame.paste(tb,(x0, y0 + tb.height))

        output_frame.save(img_bytes, format="PNG", compress_level=2)
        return img_bytes
    
    def process(self):
        video_frame_length = self.video_handler.frame_length
        audio_frame_length = self.audio_matcher.frame_length

        self.output_frame_count = int(len(self.audio_matcher.best_matches) * audio_frame_length / video_frame_length)
        
        best_matches = self.audio_matcher.get_best_matches()
        basis_coefficients = self.audio_matcher.get_basis_coefficients()

        for video_frame_i in range(self.video_handler.output_frame_count):
            elapsed_time = video_frame_i * video_frame_length
            audio_frame_i = int(elapsed_time / audio_frame_length)
            time_past_start_of_audio_frame = elapsed_time - (audio_frame_i * audio_frame_length)
            match_row = best_matches[audio_frame_i]
            match_row = [int((i * audio_frame_length + time_past_start_of_audio_frame)/video_frame_length) for i in match_row]
            
            computed_frame_bytes = self.tesselate_composite(match_row, basis_coefficients[audio_frame_i], video_frame_i)
            self.video_handler.write_frame(video_frame_i, computed_frame_bytes)
