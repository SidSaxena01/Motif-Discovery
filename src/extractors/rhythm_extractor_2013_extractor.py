# rhythm_extractor_2013_extractor.py

import essentia.standard as es
import numpy as np

from .tempo_extractor import TempoExtractor


class EssentiaTempoExtractor(TempoExtractor):
    """
    Tempo extraction using Essentia's RhythmExtractor2013.
    """

    def __init__(self, method="multifeature"):
        self.method = method

    def extract_tempo(self, audio: np.ndarray, sr: int):
        rhythm_extractor = es.RhythmExtractor2013(method=self.method)
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)

        return bpm, beats, beats_confidence
