# python src/processing_pipeline.py

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from utils.utils import (detect_motif_in_track, plot_motif_detection_results,
                         plot_star_wars_motif_matches)

JSON_MAPPING_FILE = "star_wars_comprehensive_mapping_hybrid.json"


def load_mapping_file(json_file: str) -> Dict[Any, Any]:
    with open(json_file, "r") as f:
        return json.load(f)


def get_motif_matches(
    mass_results: Dict[Any, Any],
    top_k: int = 5,
    fig_output_folder: str = "figures",
    fig_output_filename: str = "motif_matches.png",
) -> Dict[Any, Any]:
    sw_pitch = mass_results["audio_pitch"]
    sw_pitch_times = mass_results["audio_pitch_times"]
    distance_profile = mass_results["dist_profile"]  # for the motif vs. audio
    motif_hz_resampled_array = mass_results["motif_pitch"]
    motif_length = len(motif_hz_resampled_array)

    # 3) Find the top matches (exclusion zone approach)
    k = top_k
    exclusion_zone = motif_length // 2  # half the motif length
    working_distances = distance_profile.copy()

    match_indices = []
    match_distances = []

    for i in range(k):
        idx = np.argmin(working_distances)
        match_indices.append(idx)
        match_distances.append(distance_profile[idx])

        exclusion_start = max(0, idx - exclusion_zone)
        exclusion_end = min(len(working_distances), idx + exclusion_zone)
        working_distances[exclusion_start:exclusion_end] = np.inf

    # 4) Finally, call your special plotting function
    plot_star_wars_motif_matches(
        pitch_array=sw_pitch,
        pitch_times=sw_pitch_times,
        match_indices=match_indices,
        match_distances=match_distances,
        motif_length=motif_length,
        top_k=k,  # or whatever number of top matches to highlight
        out_folder=fig_output_folder,
        out_filename=fig_output_filename,
        show=False,
    )


def check_if_file_exists(filepath: str) -> bool:
    return Path(filepath).exists()

def process_audio_file(
    input_audio_filepath: str,
    input_musicxml_filepath: str,
    output_filepath: str,
    match_top_k: int = 5,
    stump_window_size: int = 2048,
) -> Dict[Any, Any]:
    
    if not check_if_file_exists(input_audio_filepath):
        raise FileNotFoundError(f"File not found: {input_audio_filepath}")
    if not check_if_file_exists(input_musicxml_filepath):
        raise FileNotFoundError(f"File not found: {input_musicxml_filepath}")

    bpm_estimate = 120
    # print(f"Processing audio file: {input_audio_filepath} vs. {input_musicxml_filepath}")
    mass_results = detect_motif_in_track(
        input_audio_filepath, input_musicxml_filepath, bpm_estimate, method="mass"
    )
    match_results = detect_motif_in_track(
        input_audio_filepath,
        input_musicxml_filepath,
        bpm_estimate,
        method="match",
        top_k=match_top_k,
    )
    stump_results = detect_motif_in_track(
        input_audio_filepath,
        input_musicxml_filepath,
        bpm_estimate,
        method="stump",
        window_size=stump_window_size,
    )

    # plot_motif_detection_results(results=mass_results, method="mass")
    # plot_motif_detection_results(results=match_results, method="match")
    # plot_motif_detection_results(results=stump_results, method="stump")

    timestamps_results = get_motif_matches(
        mass_results=mass_results,
        top_k=match_top_k,
        fig_output_folder="figures",
        fig_output_filename=f"{output_filepath}_motif_matches.png",
    )

    return {
        "input_audio_filepath": input_audio_filepath,
        "output_filepath": output_filepath,
    }


def processing_pipeline(
    show_missing_data=False,
    json_mapping_file: str = JSON_MAPPING_FILE,
    output_folder: str = "output_folder",
    output_csv: str = "processing_pipeline_results.csv",
):
    motif_mapping = load_mapping_file(json_mapping_file)

    results = []
    skiped_tracks = []

    for motif_data in motif_mapping:
        motif_id = motif_data["motif_id"]
        motif_xml_path = motif_data["musicxml"]

        for track in motif_data["tracks"]:
            track_name = track.get("track_name", None)
            track_audio_path = track.get("file_path", None)
            if track_name is None or track_audio_path is None:
                if show_missing_data:
                    print("=" * 120)
                    print(f"Skipping motif for missing info: {motif_data}")
                skiped_tracks.append(track)
                continue
            result = process_audio_file(
                input_audio_filepath=track_audio_path,
                input_musicxml_filepath=motif_xml_path,
                output_filepath=f"{motif_id}_{track_name}_output.wav",
            )
            results.append(result)

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"{len(results)} results saved to {output_csv}")
    print(f"{len(skiped_tracks)} tracks were skipped due to missing data.")


if __name__ == "__main__":
    processing_pipeline()
