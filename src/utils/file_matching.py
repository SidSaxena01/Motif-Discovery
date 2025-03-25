import argparse
import json
import os
import random
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)

# Rate limiting configuration
MIN_DELAY = 1.0  # Minimum delay between API calls in seconds
MAX_RETRIES = 5  # Maximum number of retries for rate limit errors
BACKOFF_FACTOR = 2  # Exponential backoff factor


def load_motifs_from_json(json_path):
    """Load motifs and track names from a JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def scan_audio_files(directory):
    """Scan a directory recursively for audio files."""
    audio_extensions = [".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a"]
    audio_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))

    return audio_files


def scan_musicxml_files(directory):
    """Scan a directory recursively for MusicXML files."""
    xml_extensions = [".xml", ".musicxml", ".mxl"]
    xml_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in xml_extensions):
                xml_files.append(os.path.join(root, file))

    return xml_files


def call_api_with_retry(prompt: str, model: str = "gemini-2.0-flash") -> str:
    """Call the Gemini API with retry logic for rate limiting."""
    retries = 0
    delay = MIN_DELAY

    while retries <= MAX_RETRIES:
        try:
            # Add a small delay before each API call to respect rate limits
            time.sleep(delay)

            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            return response.text.strip()

        except ClientError as e:
            # Extract the status code from the ClientError object
            # ClientError doesn't have a status_code attribute directly
            error_message = str(e)
            is_rate_limit_error = (
                "429" in error_message or "RESOURCE_EXHAUSTED" in error_message
            )

            if is_rate_limit_error:  # Rate limit exceeded
                retries += 1
                if retries > MAX_RETRIES:
                    print(f"Maximum retries ({MAX_RETRIES}) exceeded. Giving up.")
                    return "No matches found due to rate limiting."

                # Calculate backoff time with some randomness
                wait_time = delay * (1 + random.random() * 0.1)

                # Extract retry delay from error message if available
                if "retryDelay" in error_message:
                    try:
                        retry_info = error_message.split("retryDelay")[-1].split("'")[1]
                        if "s" in retry_info:
                            suggested_delay = int(retry_info.replace("s", ""))
                            wait_time = max(wait_time, suggested_delay)
                    except (IndexError, ValueError):
                        # If we can't parse the retry delay, use the calculated one
                        pass

                print(
                    f"Rate limit exceeded. Retrying in {wait_time:.1f} seconds (attempt {retries}/{MAX_RETRIES})"
                )
                time.sleep(wait_time)

                # Increase the delay for the next retry
                delay *= BACKOFF_FACTOR
            else:
                # For other types of errors, print and return empty result
                print(f"API error: {e}")
                return "No matches found due to API error."

    return "No matches found after multiple retries."


def match_motif_to_musicxml(motif_name, xml_files):
    """Match a motif name to MusicXML files using Gemini."""
    # Format the prompt for the LLM
    filenames_text = "\n".join(xml_files)
    prompt = f"""
You are an expert in matching text strings to filenames. Your task is to accurately map musical motif names to corresponding MusicXML files.

Here are the constraints:
1. Near-Exact Matches: Prioritize near-exact matches. Slight variations in spacing, capitalization, or common file extensions are acceptable, but significant differences are not.
2. Duplicate Files: The directory may contain duplicate files. Include ALL matching filepaths in the result.
3. Only consider MusicXML files (.xml, .musicxml, .mxl).
4. Case Insensitivity: Treat the motif names and filenames as case-insensitive.
5. Return full filepaths.
6. Return EACH match on a separate line with NO additional text or explanations.

Motif Name to match: {motif_name}
MusicXML Filenames: 
{filenames_text}

Output ONLY the matching filepaths, one per line. If no matches, output "No matches found."
"""

    # Call Gemini with retry logic
    return call_api_with_retry(prompt)


def match_tracks_to_audio(track_names, audio_files):
    """Match track names to audio files using Gemini."""
    if not track_names or not audio_files:
        return "No matches found."

    # Join track names for the prompt
    track_names_text = "\n".join(track_names)
    filenames_text = "\n".join(audio_files)

    prompt = f"""
You are an expert in matching text strings to filenames. Your task is to accurately map track names to corresponding audio files.

Here are the constraints:
1. Near-Exact Matches: Prioritize near-exact matches. Slight variations in spacing, capitalization, or common file extensions are acceptable, but significant differences are not.
2. Duplicate Files: The directory may contain duplicate files. Include ALL matching filepaths in the result.
3. Only consider audio files (.mp3, .wav, .flac, .ogg, .aac, .m4a).
4. Case Insensitivity: Treat the track names and filenames as case-insensitive.
5. Return full filepaths.
6. Return EACH match on a separate line with NO additional text or explanations.

Track Names to match:
{track_names_text}

Audio Filenames: 
{filenames_text}

Output ONLY the matching filepaths, one per line. If no matches, output "No matches found."
"""

    # Call Gemini with retry logic
    return call_api_with_retry(prompt)


def process_matches(matches_text):
    """Process and deduplicate matches."""
    match_list = []

    if "No matches found" not in matches_text:
        # Split by newlines and filter out empty lines or non-path entries
        match_list = [
            line.strip()
            for line in matches_text.split("\n")
            if line.strip() and os.path.exists(line.strip())
        ]

        # Remove potential duplicates while preserving order
        unique_matches = []
        for match in match_list:
            if match not in unique_matches:
                unique_matches.append(match)
        match_list = unique_matches

    return match_list


def extract_track_names_from_motif(motif_data):
    """Extract track names from a motif's YouTube links."""
    track_names = []

    if isinstance(motif_data, dict) and "youtube_links" in motif_data:
        for link in motif_data["youtube_links"]:
            if "track_name" in link and link["track_name"]:
                track_names.append(link["track_name"])

    return track_names


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load results from a checkpoint file if it exists."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(
                f"Warning: Checkpoint file {checkpoint_path} is corrupted. Starting fresh."
            )
    return {}


def save_checkpoint(results: Dict[str, Any], checkpoint_path: str):
    """Save current results to a checkpoint file."""
    with open(checkpoint_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Checkpoint saved to {checkpoint_path}")


def extract_filename(file_path):
    """Extract just the filename from a file path."""
    return os.path.basename(file_path) if file_path else None


def extract_film_code(track_name):
    """Extract film code from track name."""
    if not track_name:
        return None

    # Common patterns in track names to identify film
    patterns = [
        (r"episode\s*i\b|phantom\s*menace", "I"),
        (r"episode\s*iv\b|new\s*hope", "IV"),
        (r"episode\s*v\b|empire\s*strikes", "V"),
        (r"episode\s*vi\b|return\s*of\s*the\s*jedi", "VI"),
        (r"episode\s*vii\b|force\s*awakens", "VII"),
        (r"episode\s*viii\b|last\s*jedi", "VIII"),
        (r"episode\s*ix\b|rise\s*of\s*skywalker", "IX"),
        (r"rogue\s*one", "R"),
        (r"solo", "S"),
        (r"episode\s*ii\b|attack\s*of\s*the\s*clones", "II"),
        (r"episode\s*iii\b|revenge\s*of\s*the\s*sith", "III"),
    ]

    track_name_lower = track_name.lower()

    for pattern, code in patterns:
        if re.search(pattern, track_name_lower):
            return code

    # Default to first film if no match found
    return None


def find_best_audio_match(track_name, audio_files):
    """Find the best matching audio file for a track name."""
    if not track_name or not audio_files:
        return None

    track_name_lower = track_name.lower()
    best_match = None

    # Try to find direct matches first
    for file_path in audio_files:
        file_name = os.path.basename(file_path).lower()
        # Simple matching logic - could be enhanced with fuzzy matching
        if any(part in file_name for part in track_name_lower.split()):
            best_match = file_path
            break

    return best_match


def normalize_string(text):
    """Normalize a string for better matching."""
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove common filler words and special formatting
    replacements = [
        (r'from ".*?"', ""),  # Remove "From XYZ" parts
        (r"/audio only\)", ""),  # Remove "Audio Only" indicators
        (r"soundtrack", ""),  # Remove "soundtrack" mentions
        (r"\(\d{4}\)", ""),  # Remove years in parentheses
        (r"star wars", ""),  # Remove "Star Wars" mentions (common in all)
        (r"john williams", ""),  # Remove composer name
        (r"michael giacchino", ""),  # Remove composer name
        (r"episode [ivx]+", ""),  # Remove episode references
        (r'[\(\)\[\]\{\}"]', ""),  # Remove brackets and quotes
        (r"[-_\.]", " "),  # Replace separators with spaces
        (r"\s+", " "),  # Normalize whitespace
    ]

    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)

    return text.strip()


def calculate_similarity(s1, s2):
    """Calculate normalized similarity between two strings."""
    # Normalize strings first
    s1_norm = normalize_string(s1)
    s2_norm = normalize_string(s2)

    if not s1_norm or not s2_norm:
        return 0.0

    # Calculate similarity score using SequenceMatcher
    return SequenceMatcher(None, s1_norm, s2_norm).ratio()


def find_audio_matches(track_name, audio_files, threshold=0.5, max_matches=3):
    """Find multiple potential matches for a track name using fuzzy matching."""
    if not track_name or not audio_files:
        return []

    # Extract track number if present
    track_num_match = re.search(r"(\d{1,2})\s*[-\.]\s*", track_name)
    track_num = track_num_match.group(1) if track_num_match else None

    # Calculate similarity scores for all audio files
    matches = []
    for file_path in audio_files:
        file_name = os.path.basename(file_path)

        # Check for direct track number match which is a strong signal
        if track_num and re.search(rf"^{track_num}\s*[-_\.]\s*", file_name):
            similarity = 0.8  # Give a high base score for track number matches
        else:
            similarity = 0.0

        # Add fuzzy text similarity
        similarity += calculate_similarity(track_name, file_name) * 0.8

        # Add path similarity (for folder structure hints)
        folder = os.path.basename(os.path.dirname(file_path)).lower()
        similarity += calculate_similarity(track_name, folder) * 0.2

        # If we're reasonably confident, add to matches
        if similarity >= threshold:
            matches.append((file_path, similarity))

    # Sort by similarity score in descending order
    matches.sort(key=lambda x: x[1], reverse=True)

    # Return just the file paths (up to max_matches)
    return [m[0] for m in matches[:max_matches]]


def create_track_entry(youtube_link, audio_files):
    """Create a track entry based on YouTube link and all audio files."""
    if not youtube_link:
        return None

    track_name = youtube_link.get("track_name", "")

    # Skip tracks with empty names
    if not track_name:
        # Still create entry but with nulls for file info
        seconds = youtube_link.get("timestamp_seconds", 0)
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        timestamp = f"{minutes:02d}:{remaining_seconds:02d}"

        return {
            "track_name": "",
            "file_path": None,
            "file_name": None,
            "timestamp": timestamp,
            "timestamp_seconds": seconds,
            "youtube_url": youtube_link.get("url", ""),
            "film_code": None,
        }

    # Get possible matches with improved algorithm
    matches = find_audio_matches(track_name, audio_files)
    file_path = matches[0] if matches else None

    # Format timestamp from seconds
    seconds = youtube_link.get("timestamp_seconds", 0)
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    timestamp = f"{minutes:02d}:{remaining_seconds:02d}"

    # Get film code with fallback logic
    film_code = extract_film_code(track_name)
    if not film_code and file_path:
        # Try to extract from file path as a fallback
        film_code = extract_film_code(file_path)

    return {
        "track_name": track_name,
        "file_path": file_path,
        "file_name": extract_filename(file_path),
        "timestamp": timestamp,
        "timestamp_seconds": seconds,
        "youtube_url": youtube_link.get("url", ""),
        "film_code": film_code,
    }


def transform_to_final_format(motifs_data, results):
    """Transform results to the required output format."""
    output = []

    # Get all audio files from all motif results for comprehensive matching
    all_audio_files = []
    for result in results.values():
        if "track_matches" in result and "files" in result["track_matches"]:
            all_audio_files.extend(result["track_matches"]["files"])

    # Remove duplicates while preserving order
    unique_audio_files = []
    for file in all_audio_files:
        if file not in unique_audio_files:
            unique_audio_files.append(file)

    for motif_key, result in results.items():
        motif_data = motifs_data.get(motif_key, {})

        # Basic motif info
        entry = {
            "motif_id": motif_key,
            "motif_name": motif_data.get("name", motif_key),
            "appearances": motif_data.get("appearances", []),
            "first_appearance": motif_data.get("first_appearance", {}),
        }

        # Process tracks from YouTube links with all audio files for context
        tracks = []
        youtube_links = motif_data.get("youtube_links", [])

        for link in youtube_links:
            # Use all audio files instead of just the matches for this motif
            track = create_track_entry(link, unique_audio_files)
            if track:
                tracks.append(track)

        entry["tracks"] = tracks

        # Process MusicXML files
        musicxml_files = result.get("motif_matches", {}).get("files", [])
        if musicxml_files:
            # Take the first matching MusicXML file
            file_path = musicxml_files[0]
            entry["musicxml"] = {
                "file_path": file_path,
                "file_name": extract_filename(file_path),
            }

        output.append(entry)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Match motifs to MusicXML and tracks to audio files using Gemini LLM"
    )
    parser.add_argument(
        "--json", required=True, help="Path to the JSON file with motifs/tracks"
    )
    parser.add_argument(
        "--audio-dir", required=True, help="Directory containing audio files"
    )
    parser.add_argument(
        "--xml-dir", required=True, help="Directory containing MusicXML files"
    )
    parser.add_argument(
        "--output", help="Custom output filename", default="matching_results.json"
    )
    parser.add_argument(
        "--checkpoint", help="Checkpoint filename", default="matching_checkpoint.json"
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Index of first motif to process (0-based)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of motifs to process (0 for all)",
    )
    args = parser.parse_args()

    # Load motifs from JSON
    motifs_data = load_motifs_from_json(args.json)

    # Get list of motif keys and sort them for consistent ordering
    motif_keys = list(motifs_data.keys())

    # Determine which motifs to process
    start_idx = max(0, min(args.start, len(motif_keys) - 1))
    end_idx = (
        len(motif_keys)
        if args.limit <= 0
        else min(start_idx + args.limit, len(motif_keys))
    )
    motif_keys_to_process = motif_keys[start_idx:end_idx]

    print(
        f"Will process {len(motif_keys_to_process)} motifs (from index {start_idx} to {end_idx - 1})"
    )

    # Set up checkpoint path
    checkpoint_path = Path(args.json).parent / args.checkpoint

    # Load checkpoint if exists
    results = load_checkpoint(str(checkpoint_path))
    processed_count = sum(1 for key in motif_keys_to_process if key in results)

    if processed_count > 0:
        print(f"Found checkpoint with {processed_count} already processed motifs")

    # Scan for audio files and MusicXML files
    audio_files = scan_audio_files(args.audio_dir)
    xml_files = scan_musicxml_files(args.xml_dir)

    print(f"Found {len(audio_files)} audio files and {len(xml_files)} MusicXML files.")

    # Process each motif/track from the JSON
    try:
        for idx, motif_key in enumerate(motif_keys_to_process):
            # Skip if already processed
            if motif_key in results:
                print(f"Skipping already processed: {motif_key}")
                continue

            motif_data = motifs_data[motif_key]

            print(f"Processing {idx + 1}/{len(motif_keys_to_process)}: {motif_key}")

            # Extract track names from YouTube links in the motif data
            track_names = extract_track_names_from_motif(motif_data)
            print(f"  Using {len(track_names)} track names for audio matching")

            # 1. Match motif name to MusicXML files
            motif_matches_text = match_motif_to_musicxml(motif_key, xml_files)
            motif_matches = process_matches(motif_matches_text)

            # 2. Match track names to audio files
            track_matches_text = match_tracks_to_audio(track_names, audio_files)
            track_matches = process_matches(track_matches_text)

            # Store results
            results[motif_key] = {
                "motif_matches": {"files": motif_matches, "count": len(motif_matches)},
                "track_matches": {
                    "files": track_matches,
                    "names_used": track_names,
                    "count": len(track_matches),
                },
            }

            # Log the results
            print(f"  Found {len(motif_matches)} MusicXML matches for motif name")
            if motif_matches:
                print("    First few MusicXML matches:")
                for match in motif_matches[:3]:
                    print(f"      - {match}")
                if len(motif_matches) > 3:
                    print(f"      - ... and {len(motif_matches) - 3} more")

            print(f"  Found {len(track_matches)} audio matches for track names")
            if track_matches:
                print("    First few audio matches:")
                for match in track_matches[:3]:
                    print(f"      - {match}")
                if len(track_matches) > 3:
                    print(f"      - ... and {len(track_matches) - 3} more")
            print()

            # Save checkpoint after each motif is processed
            save_checkpoint(results, checkpoint_path)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving current progress...")
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Saving current progress before exiting...")
    finally:
        # Transform results to the specified format
        final_output = transform_to_final_format(motifs_data, results)

        # Save final results to the output file
        output_path = Path(args.json).parent / args.output
        with open(output_path, "w") as f:
            json.dump(final_output, f, indent=2)
        print(f"Results saved to {output_path}")


"""
File Matching Script Usage Instructions
======================================

This script matches musical motifs to MusicXML files and track names to audio files using the Gemini API.

Setup:
------
1. Ensure you have a Google API key for Gemini
2. Create a .env file in the project root with: GOOGLE_API_KEY=your_api_key_here
3. Install required packages: pip install python-dotenv google-generativeai

Required Arguments:
------------------
--json PATH          Path to the JSON file with motif/track data
--audio-dir PATH     Directory containing audio files (.mp3, .wav, etc.)
--xml-dir PATH       Directory containing MusicXML files (.xml, .musicxml, .mxl)

Optional Arguments:
------------------
--output FILENAME    Custom output filename (default: "matching_results.json")
--checkpoint FILENAME Checkpoint filename for resuming interrupted runs (default: "matching_checkpoint.json")
--start INDEX        Index of first motif to process, 0-based (default: 0)
--limit COUNT        Maximum number of motifs to process (default: 0 for all)

Example Commands:
----------------
# Basic usage (process all motifs):
python file_matching.py --json data/motifs.json --audio-dir data/audio --xml-dir data/musicxml

# Process only 5 motifs starting from index 10:
python file_matching.py --json data/motifs.json --audio-dir data/audio --xml-dir data/musicxml --start 10 --limit 5

# Use custom output and checkpoint filenames:
python file_matching.py --json data/motifs.json --audio-dir data/audio --xml-dir data/musicxml --output custom_results.json --checkpoint custom_checkpoint.json

Output:
-------
The script generates a JSON file with matches between motifs and MusicXML files, and between tracks and audio files.
The output includes file paths, timestamps, and other metadata helpful for source separation and analysis.
"""

if __name__ == "__main__":
    main()
