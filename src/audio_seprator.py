# audio_separator.py

import demucs.separate
from pathlib import Path


class AudioSeparator:
    """Handles audio separation using Demucs."""

    def __init__(self, target_stem: str):
        if target_stem not in ["other", "bass", "drums", "vocals", "none"]:
            raise ValueError(
                "target_stem must be one of the following: 'other', 'bass', 'drums', 'vocals' or 'none' to bypass."
            )
        self.target_stem = target_stem

    def separate_audio(self, file_path: Path) -> Path:
        """Runs Demucs to separate audio if the expected output does not already exist."""
        cleaned_filename = file_path.stem
        output_directory = Path(f"separated/htdemucs/{cleaned_filename}/")

        # Define the path for the target output file
        target_file = output_directory / f"{self.target_stem}.mp3"

        # Check if the target file already exists
        if target_file.exists():
            print(f"Target file {target_file} already exists. Skipping separation.")
            return output_directory  # Return the output directory directly

        # Run Demucs to separate audio
        demucs.separate.main(["--mp3", "-n", "htdemucs", str(file_path)])

        return output_directory

    def cleanup_files(self, output_directory: Path) -> Path:
        """Cleans up the output directory by deleting unnecessary files."""
        # Define the target file
        target_file = output_directory / f"{self.target_stem}.mp3"

        # Get all files in the output directory
        for file in output_directory.iterdir():
            if file.is_file() and file.name != target_file.name:
                file.unlink()  # Delete the unwanted file

        # Check if the target file exists
        if not target_file.exists():
            raise FileNotFoundError(f"The target file {target_file} does not exist.")

        return target_file

    def process_audio(self, filename: str) -> Path:
        """Main function to process the audio file."""
        file_path = Path(filename)

        if self.target_stem == "none":
            return file_path

        # Separate the audio
        output_directory = self.separate_audio(file_path)

        # Clean up files and get the target file path
        separated_file = self.cleanup_files(output_directory)

        return separated_file
