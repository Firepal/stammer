import numpy as np
from scipy.io import wavfile
from scipy.optimize import linear_sum_assignment
import logging

try:
    from scipy.fft import rfft as _rfft
    def _fft(x):
        return _rfft(x, workers=-1)  # multithreaded FFT
except ImportError:
    _fft = np.fft.rfft

def _to_float(audio):
    if audio.dtype == np.float32:
        return audio
    if np.issubdtype(audio.dtype, np.integer):
        intmax = np.iinfo(audio.dtype).max
        return audio.astype(np.float32) / intmax
    return audio.astype(np.float32)

class AudioMatcher:
    def __init__(self, carrier, modulator, samplerate, frame_length, chunk_size=512):
        self.carrier = _to_float(carrier)
        self.modulator = _to_float(modulator)
        assert(self.carrier.ndim == 1)
        assert(self.modulator.ndim == 1)
        self.samplerate = samplerate
        self.frame_length = frame_length
        self.chunk_size = chunk_size
        self.spectrum_band_width = 1.2

        self.samples_per_frame = int(self.frame_length * self.samplerate)
        self.window = np.hanning(self.samples_per_frame * 2).astype(np.float32)

        self.num_carrier_frames = self._count_frames(self.carrier)
        self.num_modulator_frames = self._count_frames(self.modulator)

        self.carrier_bands = None
        self.best_matches = None

        self.make_best_matches()

    def _count_frames(self, audio):
        return max(0, (len(audio) // self.samples_per_frame) - 2)

    def get_frame(self, audio, index):
        start = index * self.samples_per_frame
        end = start + self.samples_per_frame * 2
        a = audio[start:end]

        # Broadcast the 1-D window across extra dims of `a`
        win = self.window
        if a.ndim > 1:
            win = win.reshape(-1, *([1] * (a.ndim - 1)))

        frame_len = self.samples_per_frame * 2
        if len(a) < frame_len:
            pad_shape = (frame_len - len(a),) + a.shape[1:]
            a = np.concatenate([a, np.zeros(pad_shape, dtype=np.float32)])

        return (win * a).astype(np.float32, copy=False)

    def make_frames_chunk(self, audio, start_idx, end_idx):
        num = end_idx - start_idx
        frames = np.empty((num, self.samples_per_frame * 2), dtype=np.float32)
        for j, i in enumerate(range(start_idx, end_idx)):
            frames[j] = self.get_frame(audio, i)
        return frames

    def make_log_bands(self, spectra):
        split_points = [0]
        i = 2
        while i < spectra.shape[1]:
            if int(i) > split_points[-1]:
                split_points.append(int(i))
            i *= self.spectrum_band_width

        section_lengths = []
        for i in range(len(split_points) - 1):
            section_lengths.append(split_points[i+1] - split_points[i])
        section_lengths.append(spectra.shape[1] - split_points[-1])

        return np.add.reduceat(spectra, split_points, axis=1) / np.asarray(section_lengths, dtype=np.float32)

    def bands_normalized_frames(self, frames):
        transforms = _fft(frames)
        spectra = np.abs(transforms[:, 1:]).astype(np.float32, copy=False)

        bands = self.make_log_bands(spectra)

        norm = np.linalg.norm(bands, axis=1, keepdims=True)
        np.clip(norm, 1e-8, None, out=norm)
        return (bands / norm)

    def bands_normalized_chunk(self, audio, start_idx, end_idx):
        frames = self.make_frames_chunk(audio, start_idx, end_idx)
        return self.bands_normalized_frames(frames)

    def prepare_carrier_bands(self):
        if self.num_carrier_frames == 0:
            self.carrier_bands = np.empty((0, 0), dtype=np.float32)
            return
        first_end = min(self.chunk_size, self.num_carrier_frames)
        first_bands = self.bands_normalized_chunk(self.carrier, 0, first_end)
        n_bands = first_bands.shape[1]
        self.carrier_bands = np.empty((self.num_carrier_frames, n_bands), dtype=np.float32)
        self.carrier_bands[:first_end] = first_bands
        for start in range(first_end, self.num_carrier_frames, self.chunk_size):
            end = min(start + self.chunk_size, self.num_carrier_frames)
            self.carrier_bands[start:end] = self.bands_normalized_chunk(self.carrier, start, end)

    def make_best_matches(self):
        self.prepare_carrier_bands()
        self.find_matches()

    def free_match_data(self):
        del self.carrier_bands

    def find_matches(self):
        raise NotImplementedError

    def build_output_audio(self, carrier_audio):
        raise NotImplementedError

    def make_output_audio(self, _carrier_multichannel, destination_path):
        carrier = self.carrier

        channels = 1 if _carrier_multichannel.ndim == 1 else _carrier_multichannel.shape[1]
        if channels != 1:
            carrier = _to_float(_carrier_multichannel)
            output_audio = np.zeros((len(self.modulator), channels), dtype=np.float32)
        else:
            output_audio = np.zeros(len(self.modulator), dtype=np.float32)

        self.build_output_audio(output_audio, carrier, channels)
        wavfile.write(destination_path, self.samplerate, output_audio)

    def print_progress(self, _len, i):
        print(f"{int((i / _len)* 100)}%", end='     \r')


class BasicAudioMatcher(AudioMatcher):
    def basic_matching(self, c_bands_func, m_bands_func):
        self.best_matches = np.empty(self.num_modulator_frames, dtype=np.int32)
        for start in range(0, self.num_modulator_frames, self.chunk_size):
            end = min(start + self.chunk_size, self.num_modulator_frames)
            mod_chunk = m_bands_func(start, end)

            best_scores = np.full(end - start, -np.inf, dtype=np.float32)
            best_indices = np.zeros(end - start, dtype=np.int32)
            for c_start in range(0, self.num_carrier_frames, self.chunk_size):
                c_end = min(c_start + self.chunk_size, self.num_carrier_frames)
                carrier_chunk = c_bands_func(c_start, c_end)

                scores = mod_chunk @ carrier_chunk.T
                max_scores = scores.max(axis=1)
                max_idx = scores.argmax(axis=1)
                better = max_scores > best_scores
                best_scores[better] = max_scores[better]
                best_indices[better] = c_start + max_idx[better]
            self.best_matches[start:end] = best_indices
            self.print_progress(self.num_modulator_frames, (start*0.5) + (end*0.5))

    def find_matches(self):
        self.basic_matching(
            lambda start, end: self.carrier_bands[start:end],
            lambda start, end: self.bands_normalized_chunk(self.modulator, start, end)
        )

    def get_rescale_factor(self, carrier_frame, modulator_frame):
        rms_modulator = np.linalg.norm(modulator_frame)
        rms_carrier = np.linalg.norm(carrier_frame)
        if rms_carrier == 0:
            return np.zeros_like(carrier_frame)
        gain = rms_modulator / rms_carrier
        scale = gain

        frame = carrier_frame * gain
        peak = np.abs(frame).max()
        if peak > 1.0:
            scale /= peak
        return scale

    def build_output_audio(self, output_audio, carrier, channels):
        for i in range(self.num_modulator_frames):
            carrier_frame_mono = self.get_frame(self.carrier, self.best_matches[i])
            modulator_frame_mono = self.get_frame(self.modulator, i)

            scale = self.get_rescale_factor(carrier_frame_mono, modulator_frame_mono)
            if channels != 1:
                scale = scale.reshape(-1, *([1] * (channels - 1)))

                carrier_frame_multi = self.get_frame(carrier, self.best_matches[i])
                output_frame = carrier_frame_multi
            else:
                output_frame = carrier_frame_mono

            start = i * self.samples_per_frame
            end = start + self.samples_per_frame * 2
            output_audio[start:end] += output_frame * scale


class CombinedFrameAudioMatcher(AudioMatcher):
    MAX_BASIS_WIDTH = 6
    MAX_TESSELLATION_COUNT = 9

    def best_match(self, modulator_band):
        proj_indices = []
        coeffs = []
        pre, post, delta = None, None, None
        basis_epsilon = 5e-16
        while (delta is None or delta < 0) \
        and ((not coeffs) or basis_epsilon < np.abs(coeffs[-1])) \
        and ((not proj_indices) or len(proj_indices) != self.MAX_BASIS_WIDTH):
            dot_products = np.sum(self.carrier_bands * modulator_band, axis=1)
            max_idx = np.argmax(dot_products)
            proj_indices.append(max_idx)
            orth_band = self.carrier_bands[proj_indices[-1]]
            coeffs.append(np.sum(orth_band * modulator_band))
            decrement = coeffs[-1] * orth_band
            if post is not None:
                pre = post
            else:
                pre = np.sum(np.ones(len(modulator_band)) * np.abs(modulator_band))
            modulator_band -= decrement
            post = np.sum(np.ones(len(modulator_band)) * np.abs(modulator_band))
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
        self.best_matches = np.zeros((self.num_modulator_frames, self.MAX_BASIS_WIDTH), np.int32) - 1
        for start in range(0, self.num_modulator_frames, self.chunk_size):
            end = min(start + self.chunk_size, self.num_modulator_frames)
            mod_bands = self.bands_normalized_chunk(self.modulator, start, end)
            for j in range(end - start):
                i = start + j
                basis, scalars = self.best_match(mod_bands[j])
                self.best_matches[i] = basis
                self.basis_coefficients[i] = scalars
            self.print_progress(self.num_modulator_frames, (start*0.5) + (end*0.5))

    def get_carrier(self, carrier, k, c):
        composite_carrier = None
        for index, element in enumerate(c):
            if element == 0:
                break
            carrier_frame = self.get_frame(carrier, k[index])
            if index == 0:
                composite_carrier = carrier_frame * element
            else:
                composite_carrier += carrier_frame * element
        return composite_carrier

    def build_output_audio(self, output_audio, carrier, channels):
        for i in range(self.num_modulator_frames):
            composed_frame = self.get_carrier(
                carrier,
                self.best_matches[i],
                self.basis_coefficients[i]
            )
            if composed_frame is not None:
                start = i * self.samples_per_frame
                end = start + self.samples_per_frame * 2
                output_audio[start:end] += composed_frame
            self.print_progress(self.num_modulator_frames, i)


class UniqueAudioMatcher(BasicAudioMatcher):
    def find_matches(self):
        if self.num_carrier_frames < self.num_modulator_frames:
            logging.warning(f"Carrier is shorter than modulator ({self.num_carrier_frames} vs {self.num_modulator_frames}). Trimming modulator to the length of carrier")
            self.num_modulator_frames = self.num_carrier_frames

        cost_matrix = np.empty((self.num_modulator_frames, self.num_carrier_frames), dtype=np.float32)
        for start in range(0, self.num_modulator_frames, self.chunk_size):
            end = min(start + self.chunk_size, self.num_modulator_frames)
            mod_bands = self.bands_normalized_chunk(self.modulator, start, end)
            for c_start in range(0, self.num_carrier_frames, self.chunk_size):
                c_end = min(c_start + self.chunk_size, self.num_carrier_frames)
                carrier_chunk = self.carrier_bands[c_start:c_end]
                cost_matrix[start:end, c_start:c_end] = mod_bands @ carrier_chunk.T

        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
        self.best_matches = col_ind


class WeightedAudioMatcher(BasicAudioMatcher):
    # we won't be using self.carrier_bands
    def prepare_carrier_bands(self): pass

    def r_a(self, f):
        f_sq = f ** 2
        return (12194 ** 2 * f ** 4) / (
            (f_sq + 20.6 ** 2) * np.sqrt((f_sq + 107.7 ** 2) * (f_sq + 737.9 ** 2)) * (f_sq + 12194 ** 2)
        )

    def a_weighting(self, f):
        return self.r_a(f) / self.r_a(1000)

    def _weighted_chunk(self, audio, start_idx, end_idx):
        frames = self.make_frames_chunk(audio, start_idx, end_idx)
        spectra = np.abs(_fft(frames)[:, 1:]).astype(np.float32, copy=False)
        norm = np.linalg.norm(spectra, axis=1, keepdims=True)
        np.clip(norm, 1e-8, None, out=norm)
        return (spectra / norm) * self._a_weights

    def find_matches(self):
        freqs = np.fft.rfftfreq(2 * self.samples_per_frame, 1.0 / self.samplerate)[1:]
        self._a_weights = self.a_weighting(freqs).astype(np.float32, copy=False)

        self.basic_matching(
            lambda start, end: self._weighted_chunk(self.carrier, start, end),
            lambda start, end: self._weighted_chunk(self.modulator, start, end)
        )
        del self._a_weights
