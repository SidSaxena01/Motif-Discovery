import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stumpy
from dtaidistance import dtw
from music21 import chord, converter, harmony, note


def extract_midi_notes(score):
    """
    Extracts MIDI notes from a music21 score (flattened).
    Ignores chord symbols. Collects pitches of chords if found.
    """
    midi_list = []
    for el in score.flatten().notes:
        if isinstance(el, note.Note):
            midi_list.append(el.pitch.midi)
        elif isinstance(el, chord.Chord):
            for p in el.pitches:
                midi_list.append(p.midi)
        # ignoring harmony.ChordSymbol, etc.
    return midi_list


def extract_notes_with_duration(score):
    """
    Extracts pitch, duration, and offset for each Note in a music21 score.
    """
    notes_data = []
    for el in score.flatten().notes:
        if isinstance(el, note.Note):
            notes_data.append(
                {
                    "pitch": el.pitch.midi,
                    "duration": el.duration.quarterLength,
                    "offset": el.offset,
                }
            )
        elif isinstance(el, chord.Chord):
            # For a chord, you may choose to do something else or skip
            continue
    return notes_data


def midi_to_hz(midi_pitch):
    """Convert MIDI pitch to Hz."""
    return 440.0 * (2.0 ** ((midi_pitch - 69) / 12.0))


def resample_motif_to_audio_times(notes_df, audio_pitch_times, bpm):
    """
    Given a DataFrame of notes (with pitch, duration in quarter lengths),
    create a continuous time-based pitch array by resampling into a timeline
    similar to `audio_pitch_times`.
    """
    quarter_note_sec = 60.0 / bpm  # seconds per quarter note
    motif_times = []
    motif_hz = []

    current_time = 0.0
    total_audio_duration = audio_pitch_times[-1] - audio_pitch_times[0]
    # We'll assume the same sample resolution as the audio pitch
    approx_sr = (
        len(audio_pitch_times) / total_audio_duration
    )  # rough "frames per second"

    for _, row in notes_df.iterrows():
        dur_seconds = row["duration"] * quarter_note_sec
        n_samples = int(dur_seconds * approx_sr)

        if n_samples <= 0:
            continue

        # Extend the motif pitch array
        motif_times.extend(
            np.linspace(
                current_time, current_time + dur_seconds, n_samples, endpoint=False
            )
        )
        motif_hz.extend([midi_to_hz(row["pitch"])] * n_samples)

        current_time += dur_seconds

    return np.array(motif_hz, dtype=np.float64), np.array(motif_times)


def extract_pitch_essentia(audio_file):
    """
    Example placeholder for extracting pitch from an audio file with Essentia.
    Returns (pitch_array, pitch_times, sample_rate).
    """
    import essentia.standard as es
    import numpy as np

    loader = es.MonoLoader(filename=audio_file)
    audio = loader()
    sr = loader.paramValue("sampleRate")
    equalLoudness = es.EqualLoudness()
    audio_eq = equalLoudness(audio)

    pitch_extractor = es.PredominantPitchMelodia()
    pred_pitch, _ = pitch_extractor(audio_eq)

    # Filter out zero-pitch frames
    valid_mask = pred_pitch > 0
    pitch_array = pred_pitch[valid_mask]
    # Time array (Essentia typically uses 128-sample hop)
    hop_size = 128
    pitch_times = np.linspace(0.0, len(audio) / sr, len(pred_pitch))[valid_mask]

    return pitch_array, pitch_times, sr


# Optional: You could also have a separate "extract_pitch_crepe" if you prefer CREPE


def detect_motif_mass(motif_array, audio_array):
    """
    Single-query motif detection using stumpy.mass
    Returns best match index and the distance profile.
    """
    dist_profile = stumpy.mass(
        motif_array.astype(np.float64), audio_array.astype(np.float64)
    )
    best_idx = np.argmin(dist_profile)
    best_dist = dist_profile[best_idx]
    return best_idx, best_dist, dist_profile


def detect_motif_match(motif_array, audio_array, top_k=5):
    """
    Find multiple matches using stumpy.match.
    Returns an array of shape (k, 2) with [index, distance].
    """
    matches = stumpy.match(motif_array, audio_array, max_matches=top_k)
    # matches[:, 0] -> indices, matches[:, 1] -> distances
    return matches


def detect_motif_stump(audio_array, m):
    """
    Full matrix profile on audio_array using stumpy.stump (self-join).
    m is the subsequence window size.
    Returns the matrix profile and index of the top motif occurrence.
    """
    mp = stumpy.stump(audio_array, m)
    # mp[:, 0] are the profile values
    # the smallest value is the 'strongest' motif
    motif_idx = np.argmin(mp[:, 0])
    return mp, motif_idx


def detect_motif_dtw(motif_array, audio_array):
    """
    Simple DTW detection:
    We do a direct dtw.distance between motif_array and audio_array,
    but this alone won't locate the best index in the audio_array.
    Usually you'd window over audio_array in segments.
    For demonstration, we just return the distance for them as entire sequences.
    """
    dist = dtw.distance(motif_array, audio_array)
    return dist


def detect_motif_in_track(
    audio_file: str,
    score_file: str,
    bpm: float,
    method: str = "mass",
    window_size: int = 2048,
    top_k: int = 5,
):
    """
    Detects a motif (from a given musical score) in an audio track.

    Parameters
    ----------
    audio_file : str
        Path to the audio file.
    score_file : str
        Path to the MusicXML or other music21-compatible file.
    bpm : float
        Estimated or known tempo of the track in beats per minute.
    method : str
        Which detection method to use: "mass", "match", "stump", or "dtw".
    window_size : int
        Window size for the stump approach if method = "stump".
    top_k : int
        Number of matches to return if method = "match".

    Returns
    -------
    results : dict
        Various fields depending on the method chosen, typically including:
        - "indices": list or array of match indices
        - "distances": list or array of distances
        - "pitch_array": the pitch extracted from audio
        - "pitch_times": times array for the audio pitch
        - etc.
    """
    # --- 1) Parse the score
    score = converter.parse(score_file)
    notes_data = extract_notes_with_duration(score)
    notes_df = pd.DataFrame(notes_data).sort_values("offset", ascending=True)

    # --- 2) Extract pitch from the audio
    audio_pitch, audio_pitch_times, _ = extract_pitch_essentia(audio_file)

    # --- 3) Create a motif array from the score data
    motif_array, motif_times = resample_motif_to_audio_times(
        notes_df, audio_pitch_times, bpm
    )

    audio_pitch = audio_pitch.astype(np.float64)
    motif_array = motif_array.astype(np.float64)

    results = {
        "method": method,
        "audio_pitch": audio_pitch,
        "audio_pitch_times": audio_pitch_times,
        "motif_pitch": motif_array,
        "motif_times": motif_times,
    }

    # --- 4) Call the requested detection method
    if method == "mass":
        best_idx, best_dist, dist_profile = detect_motif_mass(motif_array, audio_pitch)
        results.update(
            {
                "best_idx": best_idx,
                "best_dist": best_dist,
                "dist_profile": dist_profile,
            }
        )

    elif method == "match":
        matches = detect_motif_match(motif_array, audio_pitch, top_k=top_k)
        # matches is shape (k, 2): first col = index, second col = distance
        results["matches"] = matches

    elif method == "stump":
        mp, motif_idx = detect_motif_stump(audio_pitch, window_size)
        results.update({"mp": mp, "motif_idx": motif_idx})

    elif method == "dtw":
        dtw_dist = detect_motif_dtw(motif_array, audio_pitch)
        results["dtw_distance"] = dtw_dist

    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'mass', 'match', 'stump', or 'dtw'."
        )

    return results


if __name__ == "__main__":
    # Example usage
    audio_file = "/Users/fernando/Downloads/Archive/separated/02 - Main Title-Rebel Blockade Runner_processed.wav"
    score_file = "/Users/fernando/Downloads/Archive/Xmls/1a_Main_Theme_Basic_(A_Section).musicxml"
    bpm_estimate = 145.0  # Could come from Essentia's RhythmExtractor

    # 1) Single best match with 'mass'
    mass_results = detect_motif_in_track(
        audio_file, score_file, bpm_estimate, method="mass"
    )
    print("Mass method best match index:", mass_results["best_idx"])
    print("Mass method best distance:", mass_results["best_dist"])

    # 2) Multiple matches with 'match'
    match_results = detect_motif_in_track(
        audio_file, score_file, bpm_estimate, method="match", top_k=5
    )
    print("Match top results:\n", match_results["matches"])

    # 3) Full matrix profile (self-join) with 'stump'
    stump_results = detect_motif_in_track(
        audio_file, score_file, bpm_estimate, method="stump", window_size=2048
    )
    print("Stump motif index (lowest MP value):", stump_results["motif_idx"])

    # 4) DTW distance between motif array and entire audio pitch array
    dtw_results = detect_motif_in_track(
        audio_file, score_file, bpm_estimate, method="dtw"
    )
    print("DTW distance between motif and entire track:", dtw_results["dtw_distance"])
