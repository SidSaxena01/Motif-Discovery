import json
import os
import argparse
from pydub import AudioSegment
import pandas as pd
from tqdm import tqdm

def load_results(file_path):
    """Load results from either JSON or CSV file."""
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        # Convert CSV data to structured format
        results = []
        for _, row in df.iterrows():
            # Convert string representation of dict to actual dict
            timestamps_str = row['timestamps_results']
            # Clean up the string to make it valid JSON
            timestamps_str = timestamps_str.replace("defaultdict(<class 'dict'>, ", "")
            timestamps_str = timestamps_str.rstrip(")")
            timestamps_dict = eval(timestamps_str)  # Be careful with eval
            
            results.append({
                'input_audio_filepath': row['input_audio_filepath'],
                'motif_id': row['motif_id'],
                'timestamps_results': timestamps_dict
            })
        return results
    else:
        raise ValueError("Unsupported file format. Please provide a JSON or CSV file.")

def extract_audio_segment(audio_file, start_time, end_time, output_path):
    """Extract a segment from an audio file and save it."""
    # Determine file format
    file_ext = os.path.splitext(audio_file)[1].lower()
    
    # Load audio file
    if file_ext == '.mp3':
        audio = AudioSegment.from_mp3(audio_file)
    elif file_ext == '.wav':
        audio = AudioSegment.from_wav(audio_file)
    else:
        raise ValueError(f"Unsupported audio format: {file_ext}")
    
    # Convert seconds to milliseconds
    start_ms = int(start_time * 1000)
    end_ms = int(end_time * 1000)
    
    # Extract segment
    segment = audio[start_ms:end_ms]
    
    # Save segment
    if output_path.endswith('.mp3'):
        segment.export(output_path, format='mp3')
    elif output_path.endswith('.wav'):
        segment.export(output_path, format='wav')
    else:
        segment.export(output_path, format='mp3')  # Default to mp3

def create_output_filename(input_filepath, motif_id, match_id, output_dir):
    """Create organized output file path and ensure directory exists."""
    # Get the track name from the input file path
    track_name = os.path.basename(input_filepath).split('.')[0]
    
    # Clean motif_id and track_name for use as folder names
    motif_id_clean = "".join(c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in motif_id)
    track_name_clean = "".join(c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in track_name)
    
    # Create directory structure: output_dir/motif_id/track_name/
    output_folder = os.path.join(output_dir, motif_id_clean, track_name_clean)
    os.makedirs(output_folder, exist_ok=True)
    
    # Determine file extension from input file
    file_ext = os.path.splitext(input_filepath)[1].lower()
    
    # Create output file path
    output_file = os.path.join(output_folder, f"{match_id}{file_ext}")
    return output_file

def extract_all_segments(results, output_dir):
    """Extract and save all matched segments from the results data."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for entry in tqdm(results, desc="Processing audio files"):
        audio_filepath = entry['input_audio_filepath']
        motif_id = entry['motif_id']
        timestamps = entry['timestamps_results']
        
        # Check if the audio file exists
        if not os.path.exists(audio_filepath):
            print(f"Warning: Audio file not found: {audio_filepath}")
            continue
        
        # Process each match
        for match_id, match_data in timestamps.items():
            start_time = match_data['start_time']
            end_time = match_data['end_time']
            
            # Create output file path
            output_path = create_output_filename(audio_filepath, motif_id, match_id, output_dir)
            
            try:
                extract_audio_segment(audio_filepath, start_time, end_time, output_path)
            except Exception as e:
                print(f"Error processing {audio_filepath}, {match_id}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Extract audio segments based on motif detection results')
    parser.add_argument('--input', required=True, help='Path to the JSON or CSV results file')
    parser.add_argument('--output', default='extracted_motifs', help='Output directory for extracted segments')
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.input)
    
    # Extract segments
    extract_all_segments(results, args.output)
    
    print(f"Extraction complete. Audio segments saved to: {args.output}")

if __name__ == "__main__":
    main()
