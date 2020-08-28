from __future__ import annotations

from copy import deepcopy
from math import log10
from pathlib import Path
from typing import Union

import librosa
import numpy as np
import soundfile as sf
from scipy import signal


class AudioFile:
    """
    A class for representing and modifying audio files.
    """

    def __init__(self, path: Union[str, Path] = None):
        """
        Initialize an audio file. Load file if path is provided.

        :param path: path to audio file.
        """
        self.samples: np.ndarray = np.array([])
        self.sr: int = int()
        self.filename: str = str()
        if path is not None:
            self.load(path)

    @property
    def length(self) -> int:
        """
        The length in samples of the audio data.
        """
        return self.samples.shape[-1]

    @property
    def peak(self) -> float:
        """
        The max absolute amplitude value.
        """
        max_amplitude = self.samples.max(axis=-1, initial=0.0)
        min_amplitude = self.samples.min(axis=-1, initial=0.0)
        return max(max_amplitude, abs(min_amplitude))

    @staticmethod
    def db_to_scale(db: float) -> float:
        """
        Convert a decibel value to a scalar in [0.0, 1.0]

        :param db: decibel value - 6db of gain is equivalent to scaling by 2.0
        :return: a scalar in [0.0. 1.0]
        """
        return 10.0 ** (db / 20.0)

    @staticmethod
    def scale_to_db(factor: float) -> float:
        """
        Convert a scalar value in [0.0, 1.0] to a gain adjustment in decibles.

        :param factor: a scalar in [0.0, 1.0]
        :return: decibel value - 6db of gain is equivalent to scaling by 2.0
        """
        return 20.0 * log10(factor)

    def copy(self) -> AudioFile:
        """
        Return a copy of the AudioFile.

        :return: a copy of the current object.
        """
        return deepcopy(self)

    def load(self, path: Union[str, Path]):
        """
        Load an audio file.

        :param path: path to audio file.
        """
        path = Path(path)
        if not path.exists():
            raise Exception('file {} does not exist'.format(path))
        self.samples, self.sr = librosa.load(path=path, sr=None)
        self.filename = path.name

    def save(self,
             output_path: Union[str, Path],
             filename: str = None,
             subtype: str = 'PCM_16'):
        """
        Save the current audio samples to a new file.

        :param output_path: output path for audio samples.
        :param filename: filename for output - will use the original filename of the audio file by default.
        :param subtype: soundfile subtype - 16-bit PCM by default.
        """
        if filename is None:
            filename = self.filename
        output_path = Path(output_path)
        if not output_path.exists():
            output_path.mkdir(parents=True)
        output_file = output_path.joinpath(filename)
        sf.write(output_file, self.samples, self.sr, subtype=subtype)

    def resample(self, new_sr: int):
        """
        Resample to new sample rate.

        :param new_sr: target sample rate.
        """
        self.samples = librosa.resample(self.samples, self.sr, new_sr)
        self.sr = new_sr
        return self

    def invert_polarity(self):
        """
        Invert the polarity of the waveform.
        """
        self.scale(-1)
        return self

    def scale(self, scale: float):
        """
        Scale amplitudes by a constant.

        :param scale: constant used to scale audio
        """
        self.samples = self.samples * scale
        return self

    def gain(self, db: float):
        """
        Adjust gain of audio file by decibels.

        :param db: number of decibels to adjust gain by.
        """
        scale = self.db_to_scale(db)
        self.scale(scale=scale)
        return self

    def normalize(self):
        """
        Normalize audio samples.
        """
        self.samples = librosa.util.normalize(self.samples, axis=-1)
        return self

    def clip(self, clip_db: float = 0.0):
        """
        Digitally clip audio samples by scaling beyond [-1., 1.], clipping to [-1., 1.], and reversing the scaling.

        :param clip_db: Number of decibels to clip audio by.
        """
        peak = min(1.0, self.peak)
        if peak < 1.0:
            self.normalize()
        self.gain(db=clip_db)
        np.clip(self.samples, -1.0, 1.0, out=self.samples)
        self.gain(db=-clip_db)
        self.scale(peak)
        return self

    def add_silence(self,
                    samples_before: int = None,
                    samples_after: int = None,
                    sec_before: float = 0.0,
                    sec_after: float = 0.0):
        """
        Add silence to the end of the audio samples.

        :param samples_before: number of samples of silence to add to beginning of the audio data. Takes precedence over sec_before if set.
        :param samples_after: number of samples of silence to add to end of the audio data. Takes precedence over sec_after if set.
        :param sec_before: seconds of silence to add the beginning of the audio data.
        :param sec_after: seconds of silence to add the end of the audio data.
        """
        if samples_before is None:
            samples_before = int(self.sr * sec_before)
        if samples_after is None:
            samples_after = int(self.sr * sec_after)
        silence_before = np.zeros((samples_before,), dtype=self.samples.dtype)
        self.samples = np.concatenate((silence_before, self.samples))
        silence_after = np.zeros((samples_after,), dtype=self.samples.dtype)
        self.samples = np.concatenate((self.samples, silence_after))
        return self

    def varispeed(self, rate: float):
        """
        Stretch time/pitch of audio data by increasing/decreasing playback speed.

        :param rate: Stretch factor. If rate > 1, then the signal is sped up. If rate < 1, then the signal is slowed down.
        """
        old_sr = self.sr
        self.sr *= rate
        self.resample(old_sr)
        return self

    def mix(self,
            audio: AudioFile,
            relative_start: float = 0.0,
            maintain_length: bool = False):
        """
        Combine two audio files.

        :param audio: an AudioFile to add to the current AudioFile.
        :param relative_start: the start position of audio relative to the current audio - i.e. 0.5 adds audio starting
            in the middle of the current audio.
        :param: maintain_length: if true, mixed audio will be trimmed to maintain the same length as the original audio
        """
        assert (self.sr == audio.sr)
        shape = self.samples.shape
        audio = audio.copy()

        # delay the start of the new audio
        start_sample = int(self.length * relative_start)
        audio.add_silence(samples_before=start_sample)

        # pad end of audio data if needed
        self.add_silence(samples_after=max(0, audio.length - self.length))
        audio.add_silence(samples_after=max(0, self.length - audio.length))

        # combine signals
        self.samples = self.samples + audio.samples
        if maintain_length:
            self.samples.resize(shape)
        return self

    def lpf(self, cutoff: float, order: int = 1):
        """
        Low-pass filter.

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html#scipy.signal.butter

        :param cutoff: cutoff frequency in Hz.
        :param order: the order of the filter.
        """
        sos = signal.butter(order, cutoff, btype='low', analog=False, output='sos', fs=self.sr)
        filtered = signal.sosfilt(sos, self.samples)
        self.samples = filtered
        return self

    def hpf(self, cutoff: float, order: int = 1):
        """
        High-pass filter.

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html#scipy.signal.butter

        :param cutoff: cutoff frequency in Hz.
        :param order: the order of the filter.
        """
        sos = signal.butter(order, cutoff, btype='high', analog=False, output='sos', fs=self.sr)
        filtered = signal.sosfilt(sos, self.samples)
        self.samples = filtered
        return self

    def dynamic_lpf(self,
                    cutoff: float,
                    order: int = 1,
                    relative_start: float = 0.0,
                    relative_end: float = 1.0,
                    exponential: float = 1.0):
        """
        Dynamic low-pass filter.

        :param cutoff: cutoff frequency in Hz.
        :param order: the order of the filter.
        :param relative_start: relative position to start filtering, 0.0 is the start of the audio.
        :param relative_end: relative position to end filtering, 0.0 is the end of the audio.
        :param exponential: exponential order dictates crossfade shape.
        """
        filtered = self.copy().lpf(cutoff=cutoff, order=order)

        crossfade = np.zeros(self.samples.shape)
        start_sample = int(self.length * relative_start)
        end_sample = int(self.length * relative_end)
        num = (end_sample - start_sample) // 2
        fade = np.linspace(0.000001, 1.0, num=num) ** exponential

        for i in range(fade.shape[-1]):
            crossfade[start_sample+i] = fade[i]
            crossfade[end_sample-i-1] = fade[i]

        filtered.samples = filtered.samples * crossfade
        self.samples = self.samples * (1.0 - crossfade)
        self.mix(filtered)
        return self

    def conv_reverb(self,
                    ir: AudioFile,
                    dry_db: float = 0.0,
                    wet_db: float = 0.0,
                    predelay: float = 0.0,
                    trim_tail: bool = True):
        """
        Convolution reverb.

        :param ir: AudioFile containing impulse response.
        :param dry_db: db gain to apply to dry audio.
        :param wet_db: db gain to apply to wet audio.
        :param predelay: predelay in ms added to wet audio.
        :param trim_tail: if True, will trim tail of new audio to maintain length of original audio.
        """
        ir = ir.copy()
        if ir.sr != self.sr:
            ir.resample(self.sr)

        convolution = self.copy()
        convolution.samples = signal.convolve(self.samples, ir.samples, mode='full')
        if trim_tail:
            convolution.samples.resize(self.samples.shape)

        if predelay > 0.0:
            convolution.add_silence(sec_before=predelay / 1000)

        self.gain(dry_db).mix(convolution.gain(wet_db), relative_start=0.0)
        self.clip()
        return self
