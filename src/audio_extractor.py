# audio_extractor.py

from typing import List

import librosa
import mir_eval
import soundfile as sf
from tqdm import tqdm
import numpy as np

from audio_separator import AudioSeparator
from extractors.melody_extractor import MelodyExtractor
from extractors.tempo_extractor import TempoExtractor


class AudioExtractor:
    """
    Orchestrates:
    1) Stem splitting
    2) Audio loading
    3) Melody extraction
    4) Tempo extraction
    5) Saving final results to WAV (44.1 kHz, mono)
    """

    def __init__(
        self,
        audio_separator: AudioSeparator,
        melody_extractor: MelodyExtractor,
        tempo_extractor: TempoExtractor,
        target_sr: int = 44100,
    ):
        self.audio_separator = audio_separator
        self.melody_extractor = melody_extractor
        self.tempo_extractor = tempo_extractor
        self.target_sr = target_sr

    def process_audio_file(self, input_filepath: str, output_filepath: str) -> dict:
        """
        Load the audio, run melody & tempo extractors,
        and save the processed audio as a mono WAV file (44.1 kHz).

        :param input_filepath: Path to input audio file
        :param output_filepath: Path to the resulting WAV
        :return: Dictionary containing extraction results (e.g. BPM, pitch).
        """

        # 1. Stem split the audio
        demucsed_filename = self.audio_separator.process_audio(input_filepath)

        # 2. Load audio (in mono) and resample to target_sr.
        audio, sr = librosa.load(demucsed_filename, sr=self.target_sr, mono=True)

        # 3. Melody Extraction
        pitch_values, pitch_times, pitch_confidence = (
            self.melody_extractor.extract_melody(audio, sr)
        )

        # 4. Tempo Extraction
        bpm, beats, beats_confidence = self.tempo_extractor.extract_tempo(audio, sr)

        # 5. Sonify the pitch contour
        sonification = mir_eval.sonify.pitch_contour(pitch_times, pitch_values, fs=sr)
        sf.write(output_filepath, sonification, sr, format="WAV")
        results = {
            "bpm": bpm,
            "algo": self.melody_extractor.__class__.__name__,
            # "beats": beats.tolist() if isinstance(beats, np.ndarray) else beats,
            # "beat_confidence": beats_confidence,
            "pitch_values": (
                pitch_values.tolist()
                if isinstance(pitch_values, np.ndarray)
                else pitch_values
            ),
            "pitch_times": (
                pitch_times.tolist()
                if isinstance(pitch_times, np.ndarray)
                else pitch_times
            ),
            "pitch_confidence": (
                pitch_confidence.tolist()
                if isinstance(pitch_confidence, np.ndarray)
                else pitch_confidence
            ),
        }
        return results

    def process_multiple_audio_files(
        self, input_filepaths: List[str], output_folder: str
    ) -> List[dict]:
        """
        Process multiple files in a loop.

        :param input_filepaths: List of audio files to process
        :param output_folder: Folder where the processed audio will be saved
        :return: List of result dictionaries
        """
        import os

        results_list = []

        # Create the progress bar without a specific description initially
        pbar = tqdm(input_filepaths)
        for file_path in pbar:
            # Extract filename and update the progress bar description
            filename = os.path.basename(file_path)
            pbar.set_description(f"Processing {filename}")

            # Process as before
            filename_stem = os.path.splitext(os.path.basename(file_path))[0]
            out_path = os.path.join(output_folder, f"{filename_stem}_processed.wav")
            result = self.process_audio_file(file_path, out_path)
            result.update({"file": file_path, "output_wav": out_path})
            results_list.append(result)

        return results_list
