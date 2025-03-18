# crepe_extractor.py

import crepe
import librosa
import numpy as np

from .melody_extractor import MelodyExtractor


class CrepeExtractor(MelodyExtractor):
    """
    Melody extraction using CREPE.
    """

    def __init__(
        self,
        model_capacity: str = "full",
        use_viterbi: bool = False,
        resample_sr: int = 44100,
        crepe_verbose_level: int = 1,
    ):
        self.model_capacity = model_capacity
        self.use_viterbi = use_viterbi
        self.resample_sr = resample_sr
        self.crepe_verbose_level = crepe_verbose_level

    def extract_melody(self, audio: np.ndarray, sr: int):
        """
        Uses CREPE to estimate pitch.
        Assuming audio is already loaded at the correct sampling rate.
        """
        # If sample rate differs from the expected, resample.
        if sr != self.resample_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.resample_sr)
            sr = self.resample_sr

        time, frequency, confidence, activation = crepe.predict(
            audio,
            sr=sr,
            viterbi=self.use_viterbi,
            model_capacity=self.model_capacity,
            verbose=self.crepe_verbose_level,
        )

        # # Sonify the pitch contour
        # sonification = mir_eval.sonify.pitch_contour(time, frequency, fs=sr)
        # sf.write("crepe_sonification.wav", sonification, sr, format="WAV")

        return frequency, time, confidence
