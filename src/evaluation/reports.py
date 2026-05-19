# src/evaluation/reports.py

import logging
import os
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


def analyze_timing_effects(
    voxel_scores: np.ndarray,
    feature_data: dict,
) -> dict:
    """
    Fit a linear trend to the voxel-wise prediction gains.

    Note: in the notebook this function was named analyze_timing_effects
    but does not actually use feature_data's timing — it regresses gain
    values against their own ordinal index. It is preserved faithfully here.

    Parameters
    ----------
    voxel_scores : np.ndarray, shape (n_voxels, 6)
        Output of process_voxels().
    feature_data : dict
        Annotation dict (onset → word metadata). Accepted for signature
        compatibility; not used in the regression.

    Returns
    -------
    dict with keys: slope, r_squared, mean_interval, std_interval
    """
    logger.info("Analyzing timing effects | voxel_scores shape: %s", voxel_scores.shape)

    empty = {'slope': 0.0, 'r_squared': 0.0, 'mean_interval': 0.0, 'std_interval': 0.0}

    if len(voxel_scores) == 0:
        logger.warning("No voxel scores to analyze.")
        return empty

    correlation_gains = np.array([
        score[5] for score in voxel_scores
        if not np.isnan(score[5])
    ])

    if len(correlation_gains) == 0:
        logger.warning("No valid correlation gains to analyze.")
        return empty

    X   = np.arange(len(correlation_gains)).reshape(-1, 1)
    reg = LinearRegression().fit(X, correlation_gains)

    return {
        'slope':         float(reg.coef_[0]),
        'r_squared':     float(reg.score(X, correlation_gains)),
        'mean_interval': float(np.mean(correlation_gains)),
        'std_interval':  float(np.std(correlation_gains)),
    }


def create_summary_report(
    output_dir: str,
    voxel_scores: np.ndarray,
    timing_analysis: dict,
) -> None:
    """
    Write a plain-text statistical summary of the voxel-wise encoding results.

    Sections
    --------
    1. Initial score statistics (current-word-only model)
    2. Final score statistics (current + future context model)
    3. Improvement analysis (gain = combined - current)
    4. Statistical comparison (Wilcoxon signed-rank test)
    5. Coverage statistics (total vs valid voxels)

    Parameters
    ----------
    output_dir : str
        Directory where analysis_summary.txt is written.
    voxel_scores : np.ndarray, shape (n_voxels, 6)
        Output of process_voxels(). Columns: [x,y,z, current, combined, gain].
    timing_analysis : dict
        Output of analyze_timing_effects().
    """
    os.makedirs(output_dir, exist_ok=True)

    valid_scores = voxel_scores[~np.isnan(voxel_scores).any(axis=1)]

    if len(valid_scores) == 0:
        logger.warning("No valid scores found — summary report not written.")
        return

    # Unpack score columns
    initial_correlations = valid_scores[:, 3]   # current-word brain score
    initial_r2           = valid_scores[:, 4]   # combined brain score
    correlation_gains    = valid_scores[:, 5]   # prediction gain
    final_correlations   = initial_correlations + correlation_gains

    # Linear regression on initial and final correlations
    X = np.arange(len(initial_correlations)).reshape(-1, 1)
    init_slope,  _, init_r,  init_p,  _ = stats.linregress(X.flatten(), initial_correlations)
    final_slope, _, final_r, final_p, _ = stats.linregress(X.flatten(), final_correlations)

    # Wilcoxon signed-rank test: is final > initial?
    w_stat, w_p = wilcoxon(final_correlations, initial_correlations, alternative='greater')

    improved_voxels = int(np.sum(correlation_gains > 0))

    out_path = os.path.join(output_dir, 'analysis_summary.txt')

    with open(out_path, 'w') as f:
        f.write("Brain Score Analysis Summary\n")
        f.write("===========================\n\n")

        f.write("1. Initial Score Statistics (Current Word Only):\n")
        f.write(f"   Mean Correlation: {np.mean(initial_correlations):.4f}\n")
        f.write(f"   Mean R²: {np.mean(initial_r2):.4f}\n")
        f.write(f"   Std Correlation: {np.std(initial_correlations):.4f}\n")
        f.write(f"   Regression slope: {init_slope:.4f}\n")
        f.write(f"   Regression R²: {init_r ** 2:.4f}\n\n")

        f.write("2. Final Score Statistics (With Prediction Window):\n")
        f.write(f"   Mean Correlation: {np.mean(final_correlations):.4f}\n")
        f.write(f"   Std Correlation: {np.std(final_correlations):.4f}\n")
        f.write(f"   Regression slope: {final_slope:.4f}\n")
        f.write(f"   Regression R²: {final_r ** 2:.4f}\n\n")

        f.write("3. Improvement Analysis:\n")
        f.write(f"   Mean Correlation Gain: {np.mean(correlation_gains):.4f}\n")
        f.write(f"   Std Correlation Gain: {np.std(correlation_gains):.4f}\n")
        f.write(f"   Minimum Gain: {np.min(correlation_gains):.4f}\n")
        f.write(f"   Maximum Gain: {np.max(correlation_gains):.4f}\n")
        f.write(
            f"   Relative Improvement: "
            f"{(np.mean(correlation_gains) / np.mean(initial_correlations)) * 100:.2f}%\n\n"
        )

        f.write("4. Statistical Comparison:\n")
        f.write("   Wilcoxon Signed-Rank Test Results:\n")
        f.write(f"   - W-statistic: {w_stat:.0f}\n")
        f.write(f"   - p-value: {w_p:.4e}\n\n")
        f.write("   Voxel Improvement Analysis:\n")
        f.write(f"   - Number of voxels showing improvement: {improved_voxels}\n")
        f.write(
            f"   - Percentage of voxels improved: "
            f"{(improved_voxels / len(valid_scores)) * 100:.2f}%\n\n"
        )

        f.write("5. Coverage Statistics:\n")
        f.write(f"   Total voxels analyzed: {len(voxel_scores)}\n")
        f.write(f"   Valid voxels (no NaN): {len(valid_scores)}\n")
        f.write(
            f"   Percentage valid: "
            f"{(len(valid_scores) / len(voxel_scores)) * 100:.2f}%\n"
        )

    logger.info("Summary report written to %s", out_path)