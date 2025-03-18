# melody_extractor.py

import abc

import numpy as np


class MelodyExtractor(abc.ABC):
    """
    Abstract base class that defines the interface for melody extraction.
    """

    @abc.abstractmethod
    def extract_melody(self, audio: np.ndarray, sr: int):
        """
        Extract melody (pitch contour, confidence, etc.) from an audio signal.

        :param audio: mono audio samples
        :param sr: sampling rate
        :return: (pitch_values, pitch_times, pitch_confidence)
                 or any relevant pitch extraction outputs
        """
        pass
