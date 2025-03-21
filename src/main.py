import os
import pandas as pd
import argparse

from audio_extractor import AudioExtractor
from audio_separator import AudioSeparator
from extractors.crepe_extractor import CrepeExtractor
from extractors.melodia_extractor import MelodiaExtractor
from extractors.melody_extractor import MelodyExtractor
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
                if f.lower().endswith((".wav", ".flac", ".mp3", ".aac")):
                    file_list.append(os.path.join(root, f))
        return file_list
    else:
        # Possibly comma-separated
        if "," in input_arg:
            inputs = [item.strip() for item in input_arg.split(",")]
        else:
            inputs = [input_arg.strip()]
        # Filter out invalid paths
        valid_files = [f for f in inputs if os.path.isfile(f)]
        return valid_files


def main():
    parser = argparse.ArgumentParser(
        description="Motif Detection CLI",
        usage="python main.py --input <file|file1,file2|directory> [--output <folder>]",
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

    parser.add_argument(
        "--target_stem",
        default="none",
        help="Stem separation to use. Options: 'vocals', 'bass', 'drums', 'other', or 'none' to bypass",
    )

    parser.add_argument(
        "-m",
        "--melody_extraction_method",
        default="melodia",
        help="melody extraction method to use. Options: 'melodia' and 'crepe'.",
    )
    args = parser.parse_args()

    # Make sure melody flag is correct
    if args.melody_extraction_method not in ["melodia", "crepe"]:
        raise ValueError(
            f"{args.melody_extraction_method} is not a valid melody extraction method. \
            Please choose between one of the following options: ['melodia', 'crepe']."
        )

    # Prepare input files
    input_files = collect_input_files(args.input)
    if not input_files:
        print("No valid input files found.")
        return

    # Make sure output directory exists
    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)

    # Create the AudioSeparator
    audio_separator = AudioSeparator(target_stem=args.target_stem)


    if args.melody_extraction_method == "melodia":
        # Option 1: Use Melodia for melody extraction
        melody_extractor = MelodiaExtractor(
            frame_size=2048, hop_size=128, sample_rate=44100
        )

    if args.melody_extraction_method == "crepe":
    # Option 2: Use CREPE for melody extraction (commented out for demonstration)
        melody_extractor = CrepeExtractor(
            model_capacity="full",
            use_viterbi=True,
            resample_sr=16000,
            crepe_verbose_level=1,
        )

    # Create tempo extractor
    tempo_extractor = EssentiaTempoExtractor(method="multifeature")

    # Create the orchestrator
    audio_extractor = AudioExtractor(
        audio_separator=audio_separator,
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
