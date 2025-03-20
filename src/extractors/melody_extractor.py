# melody_extractor.py

import abc

import numpy as np
import essentia.standard as es


class MelodyExtractor(abc.ABC):
    """
    Abstract base class that defines the interface for melody extraction.
    """

    equal_loudness_fn = es.EqualLoudness()

    @abc.abstractmethod
    def extract_melody(self, audio: np.ndarray, sr: int):
        """
        Extract melody (pitch contour, confidence, etc.) from an audio signal.

        :param audio: mono audio samples
        :param sr: sampling rate
        :return: (pitch_values, pitch_times, pitch_confidence)
                 or any relevant pitch extraction outputs
        """
        raise NotImplementedError("I'm just the base class bro")

    def apply_equal_loudness(self, audio: np.ndarray, sr: int):
        """
        Apply equal loudness to an audio signal
        """
        self.equal_loudness_fn.configure(sampleRate=sr)
        return self.equal_loudness_fn(audio)
