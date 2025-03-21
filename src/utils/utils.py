import os
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


def plot_motif_detection_results(
    results: dict,
    method: str,
    out_folder: str = "figures",
    fig_prefix: str = "motif_detection",
    show_plot: bool = True
):
    """
    Plots and saves a figure showing the motif detection result for a given method.
    
    Parameters
    ----------
    results : dict
        The dictionary returned by detect_motif_in_track (or a similar pipeline),
        containing at least:
        - "audio_pitch"
        - "audio_pitch_times"
        - "motif_pitch"
        - "motif_times"
        Also method-specific fields, e.g.:
          * mass -> "best_idx", "best_dist", "dist_profile"
          * match -> "matches"
          * stump -> "mp", "motif_idx"
          * dtw -> "dtw_distance"
    method : str
        Which method's result to plot: "mass", "match", "stump", or "dtw".
    out_folder : str
        Folder to save figures into (created if it doesn’t exist).
    fig_prefix : str
        Prefix for the saved figure’s file name, e.g. "motif_detection_mass.png"
    show_plot : bool
        Whether to display the plot (e.g. in a notebook). If False, just saves to file.
    """
    # Ensure output folder exists
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    audio_pitch = results["audio_pitch"]
    audio_times = results["audio_pitch_times"]
    motif_pitch = results["motif_pitch"]     # the motif pitch array used
    # motif_times = results["motif_times"]   # not always needed visually

    plt.figure(figsize=(12, 6))

    # Plot the main pitch contour
    plt.plot(audio_times, audio_pitch, label="Audio Pitch", color='gray', alpha=0.6)
    plt.title(f"Motif Detection via {method.upper()}", fontsize=14)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.grid(True, alpha=0.3)

    if method == "mass":
        # We expect "best_idx", "best_dist", "dist_profile" in results
        best_idx = results["best_idx"]
        best_dist = results["best_dist"]
        dist_profile = results["dist_profile"]

        # Compute match endpoints
        motif_length = len(motif_pitch)
        end_idx = min(best_idx + motif_length, len(audio_pitch))
        
        # Highlight the match
        match_times = audio_times[best_idx:end_idx]
        match_pitch = audio_pitch[best_idx:end_idx]
        plt.plot(match_times, match_pitch, label=f"Best match (dist={best_dist:.2f})", linewidth=2)

        # (Optional) Add a second subplot for the distance profile
        fig = plt.gcf()
        ax_pitch = plt.gca()
        fig.subplots_adjust(hspace=0.4)
        # Create an inset or second axis
        ax_dist = fig.add_axes([0.65, 0.55, 0.25, 0.3])  # x, y, width, height in [0..1]
        ax_dist.set_title("Distance Profile")
        ax_dist.plot(dist_profile, color='orange')
        ax_dist.set_xlabel("Index")
        ax_dist.set_ylabel("Distance")

    elif method == "match":
        # We expect "matches" in results; shape (k, 2) -> index, distance
        matches = results["matches"]
        motif_length = len(motif_pitch)

        # Plot top-k matches
        for i, (idx, dist) in enumerate(matches):
            idx = int(idx)
            end_idx = min(idx + motif_length, len(audio_pitch))
            match_times = audio_times[idx:end_idx]
            match_pitch = audio_pitch[idx:end_idx]

            # Use a different color each time, but a simple approach is cycle from the colormap
            plt.plot(match_times, match_pitch, linewidth=2, label=f"Match {i+1} (dist={dist:.2f})")

    elif method == "stump":
        # We expect "mp" (the matrix profile) and "motif_idx" in results
        mp = results["mp"]
        motif_idx = results["motif_idx"]   # location of the discovered motif
        motif_length = len(motif_pitch)

        # Highlight the discovered motif
        end_idx = min(motif_idx + motif_length, len(audio_pitch))
        match_times = audio_times[motif_idx:end_idx]
        match_pitch = audio_pitch[motif_idx:end_idx]
        plt.plot(match_times, match_pitch, linewidth=3, label="Discovered Motif")

        # (Optional) Plot the matrix profile in a second axes or inset
        fig = plt.gcf()
        ax_pitch = plt.gca()
        fig.subplots_adjust(hspace=0.4)
        ax_mp = fig.add_axes([0.65, 0.55, 0.25, 0.3])
        ax_mp.set_title("Matrix Profile")
        ax_mp.plot(mp[:, 0], label="Profile")
        ax_mp.set_xlabel("Index")
        ax_mp.set_ylabel("Distance")

    elif method == "dtw":
        # We expect "dtw_distance" in results
        dtw_distance = results["dtw_distance"]
        plt.text(
            0.05, 0.9,
            f"DTW Distance: {dtw_distance:.2f}",
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.7)
        )
        # For a direct DTW over entire arrays, we don’t necessarily have an index
        # unless you do a "sliding" approach. So we just annotate the distance.

    plt.legend(loc="upper right")
    save_path = os.path.join(out_folder, f"{fig_prefix}_{method}.png")
    plt.savefig(save_path, dpi=150)
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    print(f"[{method.upper()}] Figure saved to: {save_path}")

def plot_all_methods_combined(
    mass_results: dict,
    match_results: dict,
    stump_results: dict,
    dtw_results: dict,
    out_folder: str = "figures",
    fig_name: str = "motif_detection_all_methods.png",
    show_plot: bool = True
):
    """
    Creates a multi-subplot figure showing each method's detection result side-by-side.
    Saves one single figure.
    """
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # 2x2 grid
    # Flatten axes for easy indexing
    ax_list = axs.ravel()

    # Prepare universal data
    # (Assuming all results are from the same audio+motif, which they should be)
    audio_pitch = mass_results["audio_pitch"]
    audio_times = mass_results["audio_pitch_times"]

    # ---- 1) MASS subplot
    ax = ax_list[0]
    ax.plot(audio_times, audio_pitch, label="Audio Pitch", color='gray', alpha=0.6)
    best_idx = mass_results["best_idx"]
    best_dist = mass_results["best_dist"]
    motif_length = len(mass_results["motif_pitch"])
    end_idx = min(best_idx + motif_length, len(audio_pitch))
    ax.plot(audio_times[best_idx:end_idx], audio_pitch[best_idx:end_idx],
            label=f"Match (dist={best_dist:.2f})", linewidth=2)
    ax.set_title("MASS")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ---- 2) MATCH subplot
    ax = ax_list[1]
    ax.plot(audio_times, audio_pitch, label="Audio Pitch", color='gray', alpha=0.6)
    for i, (idx, dist) in enumerate(match_results["matches"]):
        idx = int(idx)
        end_idx = min(idx + motif_length, len(audio_pitch))
        ax.plot(audio_times[idx:end_idx], audio_pitch[idx:end_idx],
                linewidth=2, label=f"Match {i+1} (d={dist:.2f})")
    ax.set_title("MATCH (top-k)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ---- 3) STUMP subplot
    ax = ax_list[2]
    ax.plot(audio_times, audio_pitch, label="Audio Pitch", color='gray', alpha=0.6)
    stump_mp = stump_results["mp"]
    stump_motif_idx = stump_results["motif_idx"]
    end_idx = min(stump_motif_idx + motif_length, len(audio_pitch))
    ax.plot(audio_times[stump_motif_idx:end_idx], audio_pitch[stump_motif_idx:end_idx],
            linewidth=3, label="Discovered Motif")
    ax.set_title("STUMP (Matrix Profile)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ---- 4) DTW subplot
    ax = ax_list[3]
    ax.plot(audio_times, audio_pitch, label="Audio Pitch", color='gray', alpha=0.6)
    dtw_distance = dtw_results["dtw_distance"]
    ax.text(
        0.03, 0.87,
        f"DTW Distance: {dtw_distance:.2f}",
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.7)
    )
    ax.set_title("DTW")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("Comparison of Motif Detection Methods", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(out_folder, fig_name)
    plt.savefig(out_path, dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"Combined figure saved to: {out_path}")


def plot_motif_with_waveform(audio_samples, sr, pitch_times, pitch_array):
    # Suppose you want two subplots: top is waveform, bottom is pitch
    duration = len(audio_samples) / sr
    time_array = np.linspace(0, duration, len(audio_samples))
    
    fig, (ax_wave, ax_pitch) = plt.subplots(2, 1, figsize=(12,8), sharex=True)

    ax_wave.plot(time_array, audio_samples, color="lightblue")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_title("Waveform")

    ax_pitch.plot(pitch_times, pitch_array, color="gray")
    ax_pitch.set_ylabel("Frequency (Hz)")
    ax_pitch.set_xlabel("Time (s)")
    ax_pitch.set_title("Pitch Contour")

    plt.tight_layout()
    plt.savefig("figures/mywaveform_plot.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    # fmt: off
    # Example usage
    audio_file = "/Users/fernando/Downloads/Archive/separated/02 - Main Title-Rebel Blockade Runner_processed.wav"
    score_file = "/Users/fernando/Downloads/Archive/Xmls/1a_Main_Theme_Basic_(A_Section).musicxml"
    bpm_estimate = 145.0  # Could come from Essentia's RhythmExtractor

    # 1) Single best match with 'mass'
    mass_results = detect_motif_in_track(audio_file, score_file, bpm_estimate, method="mass")
    print("Mass method best match index:", mass_results["best_idx"])
    print("Mass method best distance:", mass_results["best_dist"])

    # 2) Multiple matches with 'match'
    match_results = detect_motif_in_track(audio_file, score_file, bpm_estimate, method="match", top_k=5)
    print("Match top results:\n", match_results["matches"])

    # 3) Full matrix profile (self-join) with 'stump'
    stump_results = detect_motif_in_track(audio_file, score_file, bpm_estimate, method="stump", window_size=2048)
    print("Stump motif index (lowest MP value):", stump_results["motif_idx"])

    # 4) DTW distance between motif array and entire audio pitch array
    dtw_results = detect_motif_in_track(audio_file, score_file, bpm_estimate, method="dtw")
    print("DTW distance between motif and entire track:", dtw_results["dtw_distance"])

    plot_motif_detection_results(results=mass_results, method="mass")
    plot_motif_detection_results(results=match_results, method="match")
    plot_motif_detection_results(results=stump_results, method="stump")
    plot_motif_detection_results(results=dtw_results, method="dtw")

    plot_all_methods_combined(mass_results, match_results, stump_results, dtw_results)
    # fmt: on