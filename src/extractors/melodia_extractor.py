# melodia_extractor.py

import essentia.standard as es
import numpy as np

from .melody_extractor import MelodyExtractor


class MelodiaExtractor(MelodyExtractor):
    """
    Melody extraction using Essentia's PredominantPitchMelodia.
    """

    def __init__(self, frame_size=2048, hop_size=128, sample_rate=44100):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate

    def extract_melody(self, audio: np.ndarray, sr: int):
        # Ensure we’re working with the correct sample rate if needed
        # or handle resampling outside this class.
        pitch_extractor = es.PredominantPitchMelodia(
            frameSize=self.frame_size, hopSize=self.hop_size
        )
        pitch_values, pitch_confidence = pitch_extractor(audio)

        duration_seconds = len(audio) / float(sr)
        pitch_times = np.linspace(0.0, duration_seconds, len(pitch_values))

        return pitch_values, pitch_times, pitch_confidence
