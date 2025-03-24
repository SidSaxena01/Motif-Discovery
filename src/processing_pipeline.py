# python src/processing_pipeline.py

import json
import pandas as pd

JSON_MAPPING_FILE = "star_wars_comprehensive_mapping_hybrid.json"


def load_mapping_file(json_file: str) -> dict:
    with open(json_file, "r") as f:
        return json.load(f)


def process_audio_file(
    input_audio_filepath: str,
    input_musicxml_filepath: str,
    output_filepath: str,
) -> dict:
    # print(f"Processing audio file: {input_audio_filepath} vs. {input_musicxml_filepath}")
    return {
        "input_audio_filepath": input_audio_filepath,
        "output_filepath": output_filepath,
    }


def main(show_missing_data=False):
    motif_mapping = load_mapping_file(JSON_MAPPING_FILE)

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

    pd.DataFrame(results).to_csv("results.csv", index=False)
    print(f"{len(results)} results saved to results.csv")
    print(f"{len(skiped_tracks)} tracks were skipped due to missing data.")


if __name__ == "__main__":
    main()
