# main.py

import os

import pandas as pd

from audio_extractor import AudioExtractor
from extractors.crepe_extractor import CrepeExtractor
from extractors.melodia_extractor import MelodiaExtractor
from extractors.rhythm_extractor_2013_extractor import EssentiaTempoExtractor


def main():
    # Paths (update to real paths in your environment)
    input_files = [
        "/Users/fernando/dev/upf/mir/motif-detection/John Williams & London Symphony Orchestra - Star Wars - The Ultimate Digital Collection (2016 - Soundtracks) [Flac 24-44_192]/46. John Williams & London Symphony Orchestra - Episode IV - Main Title.flac",
        "/Users/fernando/dev/upf/mir/motif-detection/John Williams & London Symphony Orchestra - Star Wars - The Ultimate Digital Collection (2016 - Soundtracks) [Flac 24-44_192]/47. John Williams & London Symphony Orchestra - Episode IV - Imperial Attack.flac",
        # ...
    ]
    output_folder = "/Users/fernando/dev/upf/mir/motif-detection/output_folder"
    os.makedirs(output_folder, exist_ok=True)

    # Option 1: Use Melodia for melody extraction
    melodia = MelodiaExtractor(frame_size=2048, hop_size=128, sample_rate=44100)

    # # Option 2: Use CREPE for melody extraction
    # crepe_extractor = CrepeExtractor(
    #     model_capacity='full',
    #     use_viterbi=True,
    #     resample_sr=44100,
    #     crepe_verbose_level=1
    # )

    # Tempo extractor using Essentia
    tempo_extractor = EssentiaTempoExtractor(method="multifeature")

    # Choose which melody extractor you want at runtime:
    melody_extractor = melodia  # or crepe_extractor

    # Create the AudioExtractor orchestrator
    audio_extractor = AudioExtractor(
        melody_extractor=melody_extractor,
        tempo_extractor=tempo_extractor,
        target_sr=44100,
    )

    # Process multiple files
    results = audio_extractor.process_multiple_audio_files(
        input_filepaths=input_files, output_folder=output_folder
    )

    # Print or log the results
    for res in results:
        print(f"File: {res['file']}")
        print(f"Output WAV: {res['output_wav']}")
        print(f"Extraction Results:")
        print(f"  - BPM: {res['bpm']}")
        print(f"  - Melody Extractor: {res['algo']}")
        print("----\n")

    # Save results to a CSV file
    csv_file = os.path.join(output_folder, "results.csv")
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_folder, csv_file), index=False)
    print(f"Results saved to {csv_file}")


if __name__ == "__main__":
    main()
