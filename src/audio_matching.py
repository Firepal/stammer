import numpy as np
from scipy.io import wavfile
from scipy.optimize import linear_sum_assignment
import logging
from typing import List
from PIL import Image
import image_tiling
import fraction_bits
import io
from src.framegetter import FrameGetter
from src.video_out import VideoOut


class AudioMatcher:
    # Base Class
    def __init__(self, carrier, modulator, samplerate, frame_length):
        self.carrier = carrier
        self.modulator = modulator
        self.samplerate = samplerate # samples per second
        self.frame_length = frame_length # seconds
        self.spectrum_band_width = 1.2

        self.samples_per_frame = int(self.frame_length * self.samplerate)


        self.make_best_matches()
    
    def make_frames(self, input_audio, frame_length):
        if input_audio.dtype != float:
            intmax = np.iinfo(input_audio.dtype).max
            input_audio = input_audio.astype(float) / intmax

        # todo: this will get slightly off over time.
        num_frames = (len(input_audio) // frame_length) - 2
        
        window = np.hanning(frame_length * 2)
        frames = np.zeros((num_frames, frame_length * 2), dtype=input_audio.dtype)

        for i in range(num_frames):
            a = input_audio[i*frame_length:i*frame_length+frame_length*2]
            frames[i] = (window * a)
            
        return frames

    def make_normalized_bands(self, frames_input):
        transforms = np.fft.fft(frames_input)
        spectra = abs(transforms[:,1:len(transforms[0])//2])
        split_points = [0]
        i = 2
        while i < len(spectra[0]):
            if int(i) > split_points[-1]:
                split_points.append(int(i))
            i *= self.spectrum_band_width
        section_lengths = []
        for i in range(len(split_points) - 1):
            section_lengths.append(split_points[i+1]-split_points[i])
        section_lengths.append(len(spectra[0]) - split_points[-1])
        bands = np.divide(np.add.reduceat(spectra, split_points, axis=1), section_lengths) # average value in each band
        vector_magnitudes = np.sqrt((bands * bands).sum(axis=1))
        vector_magnitudes[vector_magnitudes==0]=1
        normalized_bands = bands / vector_magnitudes[:,None]
    
        return normalized_bands

    def find_matches(self):
        raise NotImplementedError

    def make_best_matches(self):
        """
        Matches audio.
        """
        self.carrier_frames = self.make_frames(self.carrier, self.samples_per_frame)
        self.modulator_frames = self.make_frames(self.modulator, self.samples_per_frame)

        self.carrier_bands = self.make_normalized_bands(self.carrier_frames)
        self.modulator_bands = self.make_normalized_bands(self.modulator_frames)

        self.find_matches()

    def get_best_matches(self):
        return self.best_matches

    def build_output_audio(self):
        raise NotImplementedError

    def make_output_audio(self, destination_path):
        output_audio = self.build_output_audio()

        wavfile.write(destination_path, self.samplerate, output_audio)
    
    def process_video_frame(self, video_i: int, framegetter: FrameGetter, video_out: VideoOut):
        raise NotImplementedError
    
    def print_progress(self,inv_len,i):
        print(f"{int(inv_len*i*100)}%",end='     \r')


class BasicAudioMatcher(AudioMatcher):
    def best_match(self, modulator_band):
        dot_products = np.sum(self.carrier_bands * modulator_band, axis=1)
        return np.argmax(dot_products)

    def find_matches(self):
        self.best_matches = []
        inv_len = 1.0/len(self.modulator_bands)
        for i in range(len(self.modulator_bands)):
            self.best_matches.append(self.best_match(self.modulator_bands[i]))
            self.print_progress(inv_len,i)

    def get_rescaled_frame(self, carrier_frame, modulator_frame):
        # Match RMS loudness of modulator frame
        modulator_frame_amp = np.sqrt(np.sum(modulator_frame*modulator_frame))
        carrier_frame_amp = np.sqrt(np.sum(carrier_frame*carrier_frame))
        carrier_frame_amp = np.sqrt(np.sum(carrier_frame*carrier_frame))
        if (carrier_frame_amp == 0):
            return carrier_frame * 0
        rescaled_frame = carrier_frame * (modulator_frame_amp / carrier_frame_amp)

        # Don't allow clipping
        if (max(abs(rescaled_frame))) > 1:
            rescaled_frame /= max(abs(rescaled_frame))
        return rescaled_frame


    def build_output_audio(self):
        output_audio = np.zeros(self.modulator.shape, dtype=float)

        for i in range(len(self.modulator_frames)):
            carrier_frame = self.carrier_frames[self.best_matches[i]]
            modulator_frame = self.modulator_frames[i]
            start = i * self.samples_per_frame
            end = i * self.samples_per_frame + self.samples_per_frame * 2
            output_audio[start:end] += self.get_rescaled_frame(carrier_frame, modulator_frame)
        return output_audio
    
    def process_video_frame(self, video_i: int, framegetter: FrameGetter, video_out: VideoOut):
        vid_framelen = framegetter.frame_length
        audio_framelen = self.frame_length
        
        elapsed_time = video_i * vid_framelen

        audio_frame_i = int(elapsed_time / audio_framelen)
        audio_frame_fract = elapsed_time - (audio_frame_i * audio_framelen)

        match_num = self.best_matches[audio_frame_i]

        elapsed_time_in_carrier = match_num * audio_framelen + audio_frame_fract

        frame_idx = int(elapsed_time_in_carrier / framegetter.frame_length)
        # frame_idx = min(frame_idx, int(framegetter.framecount - 1))

        video_out.write_frame(framegetter.get_frame(frame_idx))


class UniqueAudioMatcher(BasicAudioMatcher):
    def find_matches(self):
        if len(self.carrier_bands) < len(self.modulator_bands):
            logging.warning(f"Carrier is shorter than modulator ({len(self.carrier_bands)} frames vs {len(self.modulator_bands)} frames). Trimming modulator to the length of carrier")
            self.modulator_bands = self.modulator_bands[:len(self.carrier_bands)]
            self.modulator_frames = self.modulator_frames[:len(self.carrier_frames)]
        # Build a 2-d array where cell i,j contains the dot product of modulator frame i and carrier frame j
        dot = lambda row: np.sum(self.carrier_bands * row, axis = 1)
        cost_matrix = np.apply_along_axis(dot, 1, self.modulator_bands)

        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
        self.best_matches = col_ind

class CombinedFrameAudioMatcher(AudioMatcher):
    MAX_BASIS_WIDTH = 6
    MAX_TESSELLATION_COUNT = 9

    def best_match(self, modulator_band):
        proj_indices = []
        coeffs = []
        pre, post, delta = None, None, None
        basis_epsilon = 5e-16
        while (delta is None or delta < 0) and ((not coeffs) or basis_epsilon < np.abs(coeffs[-1])) and ((not proj_indices) or len(proj_indices) == 1 or len(proj_indices) != self.MAX_BASIS_WIDTH): 
            dot_products = np.sum(self.carrier_bands * modulator_band, axis=1)
            max = np.argmax(dot_products)
            proj_indices.append(max)
            orth_band = self.carrier_bands[proj_indices[-1]]
            coeffs.append(np.sum(orth_band * modulator_band))
            decrement = coeffs[-1] * orth_band
            if not post is None:
                pre = post
            else:
                pre = np.sum(np.ones(len(modulator_band))* np.abs(modulator_band))
            modulator_band -= decrement
            post = np.sum(np.ones(len(modulator_band))* np.abs(modulator_band))
            delta = post - pre
        if np.abs(proj_indices[-1]) < basis_epsilon or np.abs(coeffs[-1]) < basis_epsilon:
            proj_indices.pop()
            coeffs.pop()
        padding = [0] * (self.MAX_BASIS_WIDTH - len(proj_indices))
        proj_indices = proj_indices + padding
        basis_array = np.asarray(proj_indices, dtype=np.int32)
        return (basis_array, coeffs + padding)

    def find_matches(self):
        self.basis_coefficients = {}
        self.best_matches = np.zeros((len(self.modulator_bands), self.MAX_BASIS_WIDTH), np.int32) - np.ones((len(self.modulator_bands), self.MAX_BASIS_WIDTH), np.int32)
        for i in range(len(self.modulator_bands)):
            (basis, scalars) = self.best_match(self.modulator_bands[i])
            self.best_matches[i] = basis
            self.basis_coefficients[i]= scalars

    def get_carrier(self, k,c):
        composite_carrier = None
        for index, element in enumerate(c):
            if element == 0:
                break
            if index == 0:
                composite_carrier = self.carrier_frames[k[index]]*element
            else:
                composite_carrier += self.carrier_frames[k[index]]*element
        return composite_carrier

    def build_output_audio(self):
        output_audio = np.zeros(self.modulator.shape, dtype=float)
        inv_len = len(self.modulator_frames)
        for i in range(len(self.modulator_frames)):
            composed_frame = self.get_carrier(self.best_matches[i],self.basis_coefficients[i])
            if not composed_frame is None:
                output_audio[i*self.samples_per_frame : i*self.samples_per_frame + self.samples_per_frame*2] += composed_frame
            self.print_progress(inv_len,i)
        return output_audio

    def get_basis_coefficients(self):
        return self.basis_coefficients
    
    def tesselate_composite(framegetter, video_out, match_row, basis_coefficients):
        tiles: List[Image.Image] = []
        bits: List[List[int]] = []
        used_coeffs = [(j, coefficient) for j, coefficient in enumerate(basis_coefficients) if coefficient != 0]
        for k, coeff in used_coeffs:
            frame_num = min(match_row[k], framegetter.framecount - 1)
            tiles.append(Image.open(framegetter.get_frame(frame_num)))
            hot_bits,_ = fraction_bits.as_array(coeff)
            bits.append(hot_bits)
        tesselation = image_tiling.Tiling(height=tiles[0].height,width=tiles[0].width)
        output_frame = Image.new('RGB',(tiles[0].width, tiles[0].height))
        for m in np.arange(1,CombinedFrameAudioMatcher.MAX_TESSELLATION_COUNT):
            first_hot = next(((offset, x) for offset, x in enumerate(bits) if x[m]), None)
            if first_hot is not None:
                do_tile = tesselation.needs_tiling
                tb = tiles[first_hot[0]].copy()
                x0, y0, w, h = tesselation.get_image_placement()
                tb.thumbnail((w,h))
                output_frame.paste(tb, (x0,y0))
                if do_tile:
                    output_frame.paste(tb,(x0, y0 + tb.height))
        
        img_bytes = io.BytesIO()
        output_frame.save(img_bytes,format="PNG") 
        video_out.write_frame(img_bytes)

    def process_video_frame(self, video_i: int, framegetter: FrameGetter, video_out: VideoOut):
        audio_framelen = self.frame_length
        video_framelen = framegetter.frame_length

        elapsed_time = video_i * video_framelen
        audio_frame_i = int(elapsed_time / audio_framelen)

        time_past_start_of_audio_frame = elapsed_time - (audio_frame_i * audio_framelen)

        match_row = self.best_matches[audio_frame_i]

        match_row = [int((i * audio_framelen + time_past_start_of_audio_frame)/video_framelen) for i in match_row]

        CombinedFrameAudioMatcher.tesselate_composite(framegetter, video_out, match_row, self.basis_coefficients[audio_frame_i])
