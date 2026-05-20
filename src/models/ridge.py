# src/models/ridge.py

import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from src.models.hrf import create_modulated_events, apply_hrf_convolution

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def standardize_timeseries(timeseries: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Z-score a voxel time series.

    Returns a zero vector for flat (zero-variance) signals instead of
    raising a divide-by-zero error, so the voxel loop can handle these
    gracefully via the NaN / variance checks that follow.

    Parameters
    ----------
    timeseries : np.ndarray, shape (n_timepoints,)
    epsilon : float
        Minimum std below which the signal is treated as flat.

    Returns
    -------
    np.ndarray, shape (n_timepoints,)
    """
    mean = np.mean(timeseries)
    std  = np.std(timeseries)

    if std < epsilon:
        return np.zeros_like(timeseries)

    return (timeseries - mean) / std


def compute_correlation(model, features: np.ndarray, targets: np.ndarray) -> float:
    """
    Predict with a fitted model and return the Pearson correlation with targets.

    Guards against constant predictions or constant targets â€” both produce
    undefined correlations â€” by returning 0.0 in those cases.

    Parameters
    ----------
    model : fitted sklearn estimator with .predict()
    features : np.ndarray, shape (n_samples, n_features)
    targets : np.ndarray, shape (n_samples,)

    Returns
    -------
    float
        Pearson r in [-1, 1], or 0.0 for degenerate inputs.
    """
    predictions = model.predict(features)

    if np.allclose(predictions, predictions[0]) or np.allclose(targets, targets[0]):
        return 0.0

    return float(np.corrcoef(predictions, targets)[0, 1])


# ---------------------------------------------------------------------------
# Run start time helper
# ---------------------------------------------------------------------------

def _compute_run_start_time(internal_run_index: int, run_durations: list) -> float:
    """
    Calculate the absolute start time (in seconds) for a run, correctly
    accounting for the fact that run 4 is skipped in processing but its
    duration still elapses in real time.

    internal_run_index is 0-based over the runs that were actually loaded
    (i.e. run 4 is never in this list). The mapping to actual run numbers is:
        internal index 0 â†’ run 1
        internal index 1 â†’ run 2
        internal index 2 â†’ run 3
        internal index 3 â†’ run 5  (run 4 skipped)
        internal index 4 â†’ run 6
        ...

    Parameters
    ----------
    internal_run_index : int
        0-based index into the list returned by load_all_runs().
    run_durations : list of float
        Duration in seconds for ALL 8 runs including run 4 (index 3).

    Returns
    -------
    float
        Cumulative start time in seconds.
    """
    # Map internal index â†’ actual 1-based run number
    actual_run_num = internal_run_index + 1 if internal_run_index < 3 else internal_run_index + 2

    run_start_time = 0.0
    for j in range(actual_run_num - 1):
        run_num = j + 1
        if run_num == 4:
            # Run 4 is skipped but its duration still counts toward cumulative time
            run_start_time += run_durations[3]
        else:
            dur_idx = j if j < 3 else j - 1
            run_start_time += run_durations[dur_idx]

    return run_start_time


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def process_voxels(
    test_img,
    train_imgs: list,
    test_current_embeddings: np.ndarray,
    test_current_mask: np.ndarray,
    test_future_embeddings: np.ndarray,
    test_future_mask: np.ndarray,
    train_current_embeddings_list: list,
    train_current_mask_list: list,
    train_future_embeddings_list: list,
    train_future_mask_list: list,
    events_df: pd.DataFrame,
    roi_mask: np.ndarray,
    alpha: float,
    run_durations: list,
    n_components: int = 20,
    tr_duration: float = 2.0,
) -> np.ndarray:
    """
    Voxel-wise brain encoding model with Ridge regression.

    Pipeline (mirrors the notebook's process_voxels_ba_revised step by step):

    1.  Build a combined validity mask: TRs that have BOTH a current-word
        embedding and a future-context window.
    2.  Convert the boolean mask to TR indices.
    3.  Extract valid embeddings for the test run.
    4.  Process each training run: compute valid indices, extract embeddings,
        compute absolute start time (accounting for skipped run 4).
    5.  Concatenate all training embeddings across runs.
    6.  Fit PCA (768 â†’ n_components) on training data; transform test data
        using the SAME fitted PCA â€” no data leakage.
    7.  Apply HRF convolution (SPM + derivative + dispersion) to the
        PCA-reduced features for every training run and the test run.
    8.  Build variance map; exclude voxels below the 1st percentile.
    9.  For every valid voxel:
          a. Extract and standardize the test time series.
          b. Extract and standardize each training run's time series.
          c. Fit Ridge(alpha) on current-only features â†’ brain score.
          d. Fit Ridge(alpha) on current+future features â†’ combined score.
          e. Compute prediction gain = combined_score - current_score.
    10. Return results array of shape (n_voxels, 6):
        columns: [x, y, z, current_score, combined_score, gain]

    Parameters
    ----------
    test_img : Nifti1Image
        Held-out test run (masked by process_runs).
    train_imgs : list of Nifti1Image
        Training run images (masked by process_runs), run 4 excluded.
    test_current_embeddings : np.ndarray, shape (run_length, embedding_dim)
        TR-averaged current-word embeddings for the test run.
    test_current_mask : np.ndarray, shape (run_length,), dtype bool
        True for TRs that have at least one word embedding.
    test_future_embeddings : np.ndarray, shape (run_length, window_size * embedding_dim)
        Future-context embeddings for the test run (one window size at a time).
    test_future_mask : np.ndarray, shape (run_length,), dtype bool
        True for TRs that have a valid future window.
    train_current_embeddings_list : list of np.ndarray
        One array per training run, same shape convention as test_current_embeddings.
    train_current_mask_list : list of np.ndarray
        One boolean mask per training run.
    train_future_embeddings_list : list of np.ndarray
        One array per training run, same shape convention as test_future_embeddings.
    train_future_mask_list : list of np.ndarray
        One boolean mask per training run.
    events_df : pd.DataFrame
        Speech events table (not used in the voxel loop itself; retained for
        signature compatibility with the notebook's call site).
    roi_mask : np.ndarray, shape (x, y, z), dtype bool
        ROI voxel mask from build_roi_mask(). Combined with the variance
        filter in step 8 to get the final set of voxels to process.
    alpha : float
        Ridge regression regularisation strength (e.g. 1000.0).
    run_durations : list of float
        Duration in seconds for all 8 runs including run 4.
    n_components : int
        Number of PCA components (default 20, following Caucheteux et al.).
    tr_duration : float
        TR length in seconds (default 2.0).

    Returns
    -------
    np.ndarray, shape (n_voxels, 6)
        Columns: [x, y, z, current_score, combined_score, prediction_gain]
        Returns shape (0, 6) if no voxels could be processed.
    """
    logger.info("Starting voxel processing | alpha=%.1f | n_components=%d", alpha, n_components)

    # ------------------------------------------------------------------
    # 1. Build combined validity mask for the test run
    # ------------------------------------------------------------------
    test_valid_mask = test_current_mask & test_future_mask

    logger.info(
        "Test run mask | total TRs: %d | current: %d | future: %d | both: %d (%.1f%%)",
        len(test_current_mask),
        np.sum(test_current_mask),
        np.sum(test_future_mask),
        np.sum(test_valid_mask),
        np.sum(test_valid_mask) / len(test_valid_mask) * 100,
    )
    logger.debug(
        "Mask sample (first 50 TRs):\nTR\tCurrent\tFuture\tBoth\n%s",
        "\n".join(
            f"{i}\t{int(test_current_mask[i])}\t{int(test_future_mask[i])}\t{int(test_valid_mask[i])}"
            for i in range(min(50, len(test_current_mask)))
        )
    )

    # ------------------------------------------------------------------
    # 2. Convert boolean mask â†’ TR index list
    # ------------------------------------------------------------------
    test_valid_indices = np.where(test_valid_mask)[0]

    if len(test_valid_indices) == 0:
        raise ValueError("No valid TRs found in test data â€” check embeddings and masks.")

    gaps = np.diff(test_valid_indices)
    logger.debug(
        "Gaps between valid TRs: mean=%.2f, median=%.1f, max=%d",
        np.mean(gaps), np.median(gaps), np.max(gaps)
    )

    # ------------------------------------------------------------------
    # 3. Extract valid embeddings for the test run
    # ------------------------------------------------------------------
    test_valid_current = test_current_embeddings[test_valid_indices]
    test_valid_future  = test_future_embeddings[test_valid_indices]

    # ------------------------------------------------------------------
    # 4. Process each training run
    # ------------------------------------------------------------------
    logger.info("Processing %d training run(s)...", len(train_imgs))
    train_data_by_run = []

    for i, (run_img, run_current, run_mask, run_future, run_future_mask) in enumerate(zip(
        train_imgs,
        train_current_embeddings_list,
        train_current_mask_list,
        train_future_embeddings_list,
        train_future_mask_list,
    )):
        run_valid_mask    = run_mask & run_future_mask
        run_valid_indices = np.where(run_valid_mask)[0]

        if len(run_valid_indices) == 0:
            logger.warning("No valid TRs in training run %d â€” skipping.", i + 1)
            continue

        run_valid_current = run_current[run_valid_indices]
        run_valid_future  = run_future[run_valid_indices]

        # Absolute start time, accounting for skipped run 4
        actual_run_num = i + 1 if i < 3 else i + 2
        run_start_time = _compute_run_start_time(i, run_durations)

        logger.info(
            "Training run %d (actual run %d): start=%.2fs | valid TRs=%d",
            i + 1, actual_run_num, run_start_time, len(run_valid_indices)
        )

        train_data_by_run.append({
            'current':         run_valid_current,
            'future':          run_valid_future,
            'indices':         run_valid_indices,
            'run_index':       i,
            'actual_run_num':  actual_run_num,
            'start_time':      run_start_time,
            'img':             run_img,
        })

    if not train_data_by_run:
        raise ValueError("No valid TRs found in any training run.")

    # ------------------------------------------------------------------
    # 5. Concatenate all training embeddings
    # ------------------------------------------------------------------
    all_train_current = np.vstack([r['current'] for r in train_data_by_run])
    all_train_future  = np.vstack([r['future']  for r in train_data_by_run])

    logger.info(
        "Concatenated training embeddings | current: %s | future: %s",
        all_train_current.shape, all_train_future.shape
    )

    # ------------------------------------------------------------------
    # 6. PCA â€” fit on training, transform test (no leakage)
    # ------------------------------------------------------------------
    logger.info("Fitting PCA (n_components=%d) on training data...", n_components)

    pca_current = PCA(n_components=n_components)
    train_current_reduced = pca_current.fit_transform(all_train_current)
    logger.info(
        "Current PCA explained variance: %.1f%%",
        np.sum(pca_current.explained_variance_ratio_) * 100
    )

    pca_future = PCA(n_components=n_components)
    train_future_reduced = pca_future.fit_transform(all_train_future)
    logger.info(
        "Future PCA explained variance: %.1f%%",
        np.sum(pca_future.explained_variance_ratio_) * 100
    )

    # Transform test using the SAME fitted PCAs â€” critical to prevent leakage
    test_current_reduced = pca_current.transform(test_valid_current)
    test_future_reduced  = pca_future.transform(test_valid_future)

    # ------------------------------------------------------------------
    # 7. HRF convolution on PCA-reduced features
    # ------------------------------------------------------------------
    logger.info("Applying HRF convolution to training runs...")
    train_convolved_by_run = []

    for run_data in train_data_by_run:
        run_current_reduced = pca_current.transform(run_data['current'])
        run_future_reduced  = pca_future.transform(run_data['future'])

        current_events = create_modulated_events(
            pc_data=run_current_reduced,
            valid_indices=run_data['indices'],
            run_start_time=run_data['start_time'],
            n_components=n_components,
            prefix="current",
        )
        future_events = create_modulated_events(
            pc_data=run_future_reduced,
            valid_indices=run_data['indices'],
            run_start_time=run_data['start_time'],
            n_components=n_components,
            prefix="future",
        )

        combined_events = pd.concat([current_events, future_events], ignore_index=True)
        convolved, _ = apply_hrf_convolution(
            run_data['img'], combined_events, run_data['indices']
        )
        train_convolved_by_run.append(convolved)

        logger.info(
            "Run %d HRF convolution done | convolved shape: %s",
            run_data['actual_run_num'], convolved.shape
        )

    train_convolved = np.vstack(train_convolved_by_run)

    # Test run HRF convolution
    logger.info("Applying HRF convolution to test run...")
    test_run_index    = len(train_imgs)
    test_start_time   = _compute_run_start_time(test_run_index, run_durations)

    test_current_events = create_modulated_events(
        pc_data=test_current_reduced,
        valid_indices=test_valid_indices,
        run_start_time=test_start_time,
        n_components=n_components,
        prefix="current",
    )
    test_future_events = create_modulated_events(
        pc_data=test_future_reduced,
        valid_indices=test_valid_indices,
        run_start_time=test_start_time,
        n_components=n_components,
        prefix="future",
    )

    test_combined_events = pd.concat([test_current_events, test_future_events], ignore_index=True)
    test_convolved, _ = apply_hrf_convolution(test_img, test_combined_events, test_valid_indices)

    # Split convolved features into current-only and combined matrices
    train_features_current  = train_convolved[:, :n_components]
    train_features_combined = train_convolved

    test_features_current   = test_convolved[:, :n_components]
    test_features_combined  = test_convolved

    logger.info(
        "Feature matrices | train current: %s | train combined: %s | "
        "test current: %s | test combined: %s",
        train_features_current.shape, train_features_combined.shape,
        test_features_current.shape,  test_features_combined.shape,
    )

    # ------------------------------------------------------------------
    # 8. Variance filter â€” exclude near-zero-variance voxels
    # ------------------------------------------------------------------
    test_data      = test_img.get_fdata()
    variance_map   = np.var(test_data, axis=-1)
    var_threshold  = np.percentile(variance_map[variance_map > 0], 1)
    valid_voxels   = roi_mask & (variance_map > var_threshold)

    total_voxels = int(np.sum(valid_voxels))
    logger.info("Valid voxels after variance filter: %d", total_voxels)

    # ------------------------------------------------------------------
    # 9 & 10 & 11. Voxel loop â€” Ridge regression
    # ------------------------------------------------------------------
    run_info = [
        {'run_index': r['run_index'], 'valid_indices': r['indices']}
        for r in train_data_by_run
    ]

    voxel_results     = []
    all_current_scores = []
    all_combined_scores = []
    all_gains          = []
    skipped = {'variance': 0, 'train_data': 0, 'other': 0}

    progress_step = max(1, total_voxels // 10)

    logger.info("Starting voxel loop over %d voxels...", total_voxels)

    for processed, (x, y, z) in enumerate(zip(*np.where(valid_voxels)), start=1):

        if processed % progress_step == 0:
            logger.info(
                "Voxel progress: %d/%d (%.1f%%)",
                processed, total_voxels, processed / total_voxels * 100
            )

        # --- Test voxel time series ---
        test_ts = test_data[x, y, z, test_valid_indices]

        if np.var(test_ts) < 1e-10:
            skipped['variance'] += 1
            continue

        test_ts_std = standardize_timeseries(test_ts)

        if np.any(np.isnan(test_ts_std)):
            skipped['variance'] += 1
            continue

        # --- Training voxel time series (all runs) ---
        train_ts_parts = []

        for info in run_info:
            run_img  = train_imgs[info['run_index']]
            run_data_arr = run_img.get_fdata()
            run_ts   = run_data_arr[x, y, z, info['valid_indices']]

            if np.var(run_ts) < 1e-10:
                continue

            run_ts_std = standardize_timeseries(run_ts)

            if np.any(np.isnan(run_ts_std)):
                continue

            train_ts_parts.append(run_ts_std)

        if not train_ts_parts:
            skipped['train_data'] += 1
            continue

        train_ts_std = np.concatenate(train_ts_parts)

        # --- Ridge regression ---
        try:
            # Model 1: current word only
            model_current = Ridge(alpha=alpha)
            model_current.fit(train_features_current, train_ts_std)
            current_preds = model_current.predict(test_features_current)
            current_score = float(np.corrcoef(current_preds, test_ts_std)[0, 1])
            if np.isnan(current_score) or np.isinf(current_score):
                current_score = 0.0

            # Model 2: current + future context
            model_combined = Ridge(alpha=alpha)
            model_combined.fit(train_features_combined, train_ts_std)
            combined_preds = model_combined.predict(test_features_combined)
            combined_score = float(np.corrcoef(combined_preds, test_ts_std)[0, 1])
            if np.isnan(combined_score) or np.isinf(combined_score):
                combined_score = 0.0

            gain = combined_score - current_score

            # Verbose logging for first voxel and every 20%
            log_this = (processed == 1) or (processed % max(1, total_voxels // 5) == 0)
            if log_this:
                logger.debug(
                    "Voxel (%d,%d,%d) | current=%.4f | combined=%.4f | gain=%.4f",
                    x, y, z, current_score, combined_score, gain
                )

            all_current_scores.append(current_score)
            all_combined_scores.append(combined_score)
            all_gains.append(gain)
            voxel_results.append([x, y, z, current_score, combined_score, gain])

        except Exception as e:
            logger.warning("Error at voxel (%d,%d,%d): %s", x, y, z, e)
            skipped['other'] += 1
            continue

    # ------------------------------------------------------------------
    # Summary logging
    # ------------------------------------------------------------------
    logger.info(
        "Voxel loop complete | processed: %d | skipped variance: %d | "
        "skipped train_data: %d | skipped other: %d",
        len(voxel_results), skipped['variance'], skipped['train_data'], skipped['other']
    )

    if all_current_scores:
        logger.info(
            "Score summary | current: mean=%.4f min=%.4f max=%.4f | "
            "combined: mean=%.4f | gain: mean=%.4f | positive gain: %d/%d (%.1f%%)",
            np.mean(all_current_scores), min(all_current_scores), max(all_current_scores),
            np.mean(all_combined_scores),
            np.mean(all_gains),
            sum(g > 0 for g in all_gains), len(all_gains),
            sum(g > 0 for g in all_gains) / len(all_gains) * 100,
        )

    if not voxel_results:
        raise ValueError("No valid voxels were processed successfully.")

    return np.array(voxel_results)