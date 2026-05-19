# src/evaluation/timing.py

import logging
import os
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def create_timing_visualization(
    output_dir: str,
    feature_data: dict,
    voxel_scores: np.ndarray,
    time_threshold: float = 2.0,
) -> np.ndarray:
    """
    Scatter plot of inter-word time interval vs voxel-wise prediction gain.

    For each consecutive word pair in feature_data, computes the gap
    (next_onset - current_offset) and pairs it with the corresponding
    voxel gain score. Only intervals within time_threshold are plotted.
    A linear trend line is overlaid.

    Parameters
    ----------
    output_dir : str
        Directory where timing_analysis.png is saved.
    feature_data : dict
        Annotation dict from process_annotations(): onset → word metadata.
    voxel_scores : np.ndarray, shape (n_voxels, 6)
        Output of process_voxels(). Column 5 is prediction gain.
    time_threshold : float
        Maximum inter-word interval to include (seconds). Default 2.0.

    Returns
    -------
    np.ndarray
        The valid (filtered, NaN-free) interval values that were plotted.
        Returns empty array if nothing could be plotted.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(
        "Creating timing visualization | threshold=%.1fs | voxel_scores=%s",
        time_threshold, voxel_scores.shape
    )

    timepoints = sorted(feature_data.keys())[:-1]  # exclude last word (no successor)
    time_intervals = []
    gains          = []

    for i, current_time in enumerate(timepoints):
        if i >= len(voxel_scores):
            break

        current_onset    = feature_data[current_time]['original_onset']
        current_duration = feature_data[current_time]['duration']

        if i + 1 < len(timepoints):
            next_onset = feature_data[timepoints[i + 1]]['original_onset']
            interval   = next_onset - (current_onset + current_duration)

            if interval <= time_threshold:
                time_intervals.append(interval)
                gains.append(voxel_scores[i, 5])

    if not time_intervals:
        logger.warning("No valid timing data to plot.")
        return np.array([])

    intervals = np.array(time_intervals)
    gains     = np.array(gains)

    # Remove NaNs
    valid = ~np.isnan(intervals) & ~np.isnan(gains)
    intervals = intervals[valid]
    gains     = gains[valid]

    logger.info("Valid data points for timing plot: %d", len(intervals))

    if len(intervals) == 0:
        logger.warning("All timing data points were NaN — nothing to plot.")
        return np.array([])

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(intervals, gains, alpha=0.5, c=intervals, cmap='viridis')
    fig.colorbar(scatter, ax=ax, label='Time Interval (seconds)')

    z = np.polyfit(intervals, gains, 1)
    p = np.poly1d(z)
    ax.plot(intervals, p(intervals), 'r--', alpha=0.8, label=f'Trend (slope: {z[0]:.4f})')

    ax.set_xlabel('Time Interval Between Words (seconds)')
    ax.set_ylabel('Correlation Gain')
    ax.set_title(
        f'Relationship between Word Timing and Prediction Gains\n'
        f'(Intervals ≤ {time_threshold}s)'
    )
    ax.set_xlim(-0.1, time_threshold + 0.1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path = os.path.join(output_dir, 'timing_analysis.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Timing visualization saved to %s", out_path)
    return intervals