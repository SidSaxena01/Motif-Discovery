# tempo_extractor.py

import abc

import numpy as np


class TempoExtractor(abc.ABC):
    """
    Abstract base class that defines the interface for tempo extraction.
    """

    @abc.abstractmethod
    def extract_tempo(self, audio: np.ndarray, sr: int):
        """
        Extract tempo and (optionally) beat information from an audio signal.

        :param audio: mono audio samples
        :param sr: sampling rate
        :return: bpm (float), beat_positions (np.ndarray), confidence (float)
                 or other relevant tempo outputs.
        """
        pass
