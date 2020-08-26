from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Union

import librosa
import numpy as np
from scipy import signal
import soundfile as sf


class AudioFile:
    """
    A class for representing audio files.
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
    def length(self):
        """
        The length in samples of the audio data.
        """
        return self.samples.shape[-1]

    @property
    def peak(self):
        """
        The max absolute amplitude value.
        """
        max_amplitude = self.samples.max(axis=-1, initial=0.0)
        min_amplitude = self.samples.min(axis=-1, initial=0.0)
        return max(max_amplitude, abs(min_amplitude))

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

    def save(self, output_path: Union[str, Path], filename: str = None, subtype: str = 'PCM_16'):
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

    def scale_amplitude(self, scale: float):
        """
        Scale amplitudes by a constant.
        :param scale: constant used to scale audio
        """
        self.samples = self.samples * scale
        return self

    def normalize(self):
        """
        Normalize audio samples.
        """
        self.samples = librosa.util.normalize(self.samples, axis=-1)
        return self

    def clip_amplitude(self, clip_amount: float = 2.0):
        """
        Digitally clip audio samples by scaling beyond [-1., 1.] and clipping to [-1., 1.].
        :param clip_amount: A number > 1.0 used to scale the normalized audio samples.
        """
        peak = self.peak
        self.normalize().scale_amplitude(clip_amount)
        np.clip(self.samples, -1.0, 1.0, out=self.samples)
        self.scale_amplitude(peak / clip_amount)
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

    # def pitch_shift(self, n_steps: float):
    #     """
    #     Shift pitch of audio data.
    #     :param n_steps: how many steps to shift
    #     """
    #     self.samples = librosa.effects.pitch_shift(self.samples, self.sr, n_steps=n_steps)

    def varispeed(self, rate: float):
        """
        Stretch time/pitch of audio data by increasing/decreasing playback speed.
        :param rate: Stretch factor. If rate > 1, then the signal is sped up. If rate < 1, then the signal is slowed down.
        """
        old_sr = self.sr
        self.sr *= rate
        self.resample(old_sr)
        return self

    def mix(self, audio: AudioFile, relative_start: float = 1.0):
        """
        Combine two audio files.
        :param audio: an AudioFile to add to the current AudioFile.
        :param relative_start: the start position of audio relative to the current audio - i.e. 0.5 adds audio starting
        in the middle of the current audio.
        """
        assert(self.sr == audio.sr)
        audio = deepcopy(audio)

        # delay the start of the new audio
        start_sample = int(self.length * relative_start)
        audio.add_silence(samples_before=start_sample)

        # pad end of audio data if needed
        self.add_silence(samples_after=max(0, audio.length - self.length))
        audio.add_silence(samples_after=max(0, self.length - audio.length))

        # combine signals
        self.samples = self.samples + audio.samples
        return self

    def lpf(self, order, cutoff):
        """
        Low-pass filter. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html#scipy.signal.butter
        :param order: the order of the filter.
        :param cutoff: cutoff frequency in Hz.
        """
        sos = signal.butter(order, cutoff, btype='low', analog=False, output='sos', fs=self.sr)
        filtered = signal.sosfilt(sos, self.samples)
        self.samples = filtered
        return self

    def add_reverberation(self):
        return self
