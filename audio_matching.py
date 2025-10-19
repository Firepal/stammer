import numpy as np
from scipy.io import wavfile
from scipy.optimize import linear_sum_assignment
import logging


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
        transforms = np.fft.rfft(frames_input)
        spectra = abs(transforms[:,1:])
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
    
    def print_progress(self,inv_len,i):
        print(f"{int(inv_len*i*100)}%",end='     \r')


class BasicAudioMatcher(AudioMatcher):
    def best_match(self, modulator_band):
        dot_products = np.sum(self.carrier_bands * modulator_band, axis=1)
        return np.argmax(dot_products)

    def find_matches(self):
        scores = np.dot(self.modulator_bands.astype(np.float32, copy=False),
                        self.carrier_bands.T.astype(np.float32, copy=False))
        self.best_matches = np.argmax(scores, axis=1)

    def get_rescaled_frame(self, carrier_frame, modulator_frame):
        rms_modulator = np.linalg.norm(modulator_frame)
        rms_carrier = np.linalg.norm(carrier_frame)
        if rms_carrier == 0:
            return np.zeros_like(carrier_frame)
        gain = rms_modulator / rms_carrier
        frame = carrier_frame * gain
        # clipping
        peak = np.abs(frame).max()
        if peak > 1.0:
            frame /= peak
        return frame

    def build_output_audio(self):
        output_audio = np.zeros(self.modulator.shape, dtype=float)

        for i in range(len(self.modulator_frames)):
            carrier_frame = self.carrier_frames[self.best_matches[i]]
            modulator_frame = self.modulator_frames[i]
            start = i * self.samples_per_frame
            end = i * self.samples_per_frame + self.samples_per_frame * 2
            output_audio[start:end] += self.get_rescaled_frame(carrier_frame, modulator_frame)
        return output_audio

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

class WeightedAudioMatcher(BasicAudioMatcher):
    def r_a(self, f):
        f_sq = f**2
        return (12194**2 * f**4) / (
            (f_sq + 20.6**2) * np.sqrt((f_sq + 107.7**2) * (f_sq + 737.9**2)) * (f_sq + 12194 ** 2)
        )
    
    def a_weighting(self, f):
        return self.r_a(f) / self.r_a(1000)

    def make_normalized_bands(self, frames_input):
        spectra = np.abs(np.fft.rfft(frames_input)[:, 1:])
        norm = np.linalg.norm(spectra, axis=1, keepdims=True)
        np.clip(norm, 1e-8, None, out=norm)

        return (spectra / norm).astype(np.float32, copy=False)
    
    def find_matches(self):
        freqs = np.fft.rfftfreq(2 * self.samples_per_frame,
                                1.0 / self.samplerate)[1:]
        w = self.a_weighting(freqs).astype(np.float32, copy=False)
        weighted_carriers = self.carrier_bands.astype(np.float32) * w  # (n_frames, n_bins)
        scores = np.dot(self.modulator_bands.astype(np.float32), weighted_carriers.T)

        self.best_matches = np.argmax(scores, axis=1)

    def best_match(self, modulator_band, weights):
        dot_products = np.sum(weights * (self.carrier_bands * modulator_band), axis=1)
        best = np.argmax(dot_products)
        return best
