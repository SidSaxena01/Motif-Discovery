import os
import pandas as pd
import argparse

from audio_extractor import AudioExtractor
from extractors.crepe_extractor import CrepeExtractor
from extractors.melodia_extractor import MelodiaExtractor
from extractors.rhythm_extractor_2013_extractor import EssentiaTempoExtractor


def collect_input_files(input_arg):
    """
    Given an argument that may be a single file, multiple comma-separated files, or a directory,
    return a list of valid file paths.
    """
    # If it's a directory, return all valid files within (e.g. audio files).
    if os.path.isdir(input_arg):
        # Collect all files within the directory. Adjust filter if needed to match audio file types.
        file_list = []
        for root, _, files in os.walk(input_arg):
            for f in files:
                # Simple filter - you can add more extensions if needed
                if f.lower().endswith(('.wav', '.flac', '.mp3', '.aac')):
                    file_list.append(os.path.join(root, f))
        return file_list
    else:
        # Possibly comma-separated
        if ',' in input_arg:
            inputs = [item.strip() for item in input_arg.split(',')]
        else:
            inputs = [input_arg.strip()]
        # Filter out invalid paths
        valid_files = [f for f in inputs if os.path.isfile(f)]
        return valid_files


def main():
    parser = argparse.ArgumentParser(
        description="Motif Detection CLI",
        usage="python main.py --input <file|file1,file2|directory> [--output <folder>]"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to a single audio file, multiple comma-separated files, or a directory containing audio files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output_folder",
        help="Path to the output directory. Defaults to 'output_folder'.",
    )
    args = parser.parse_args()

    # Prepare input files
    input_files = collect_input_files(args.input)
    if not input_files:
        print("No valid input files found.")
        return

    # Make sure output directory exists
    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)

    # Option 1: Use Melodia for melody extraction
    melodia_extractor = MelodiaExtractor(frame_size=2048, hop_size=128, sample_rate=44100)

    # Option 2: Use CREPE for melody extraction (commented out for demonstration)
    crepe_extractor = CrepeExtractor(
        model_capacity='full',
        use_viterbi=True,
        resample_sr=16000,
        crepe_verbose_level=1
    )

    # Create tempo extractor
    tempo_extractor = EssentiaTempoExtractor(method="multifeature")

    # Choose which melody extractor you want at runtime:
    melody_extractor = crepe_extractor # or melodia_extractor

    # Create the orchestrator
    audio_extractor = AudioExtractor(
        melody_extractor=melody_extractor,
        tempo_extractor=tempo_extractor,
        target_sr=44100,
    )

    # Process files
    results = audio_extractor.process_multiple_audio_files(
        input_filepaths=input_files, output_folder=output_folder
    )

    # # Display results
    # for res in results:
    #     print(f"File: {res['file']}")
    #     print(f"Output WAV: {res['output_wav']}")
    #     print("Extraction Results:")
    #     print(f"  - BPM: {res['bpm']}")
    #     print(f"  - Melody Extractor: {res['algo']}")
    #     print("----\n")

    # Save CSV
    csv_file = os.path.join(output_folder, "results.csv")
    pd.DataFrame(results).to_csv(csv_file, index=False)
    print(f"Results saved to {csv_file}")

if __name__ == "__main__":
    main()
