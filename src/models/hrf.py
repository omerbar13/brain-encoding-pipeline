# src/models/hrf.py

import logging
import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel

logger = logging.getLogger(__name__)


def create_modulated_events(
    pc_data: np.ndarray,
    valid_indices: np.ndarray,
    run_start_time: float,
    n_components: int,
    prefix: str = "",
    tr_duration: float = 2.0,
) -> pd.DataFrame:
    """
    Build a nilearn-compatible modulated events DataFrame for PCA components.

    Each PC gets one event per valid TR, where the PC value at that TR
    acts as a parametric modulator of the HRF amplitude. This is how
    continuous neural signals are encoded into the GLM framework.

    For example, with n_components=20 and 150 valid TRs, this produces
    20 Ã— 150 = 3000 rows, one per (component, TR) combination.

    Parameters
    ----------
    pc_data : np.ndarray, shape (n_valid_trs, n_components)
        PCA-reduced embeddings for the valid TRs only.
    valid_indices : np.ndarray, shape (n_valid_trs,)
        TR indices (within the run) that have valid embeddings.
        Used to compute onset times: onset = tr_idx * tr_duration.
    run_start_time : float
        Absolute start time of this run in seconds. NOT added to onset
        times here â€” nilearn's FirstLevelModel expects onsets relative
        to the start of the run image being fitted, not absolute time.
    n_components : int
        Number of PCA components (must match pc_data.shape[1]).
    prefix : str
        Prepended to the trial_type name, e.g. "current" â†’ "currentpc0".
        Use "current" for current-word components and "future" for
        future-context components so they can be separated after convolution.
    tr_duration : float
        TR length in seconds (default 2.0).

    Returns
    -------
    pd.DataFrame
        Columns: trial_type (str), onset (float), duration (float),
                 modulation (float).
        Ready to pass directly to nilearn's FirstLevelModel.fit().
    """
    modulated_events = []

    for pc_idx in range(n_components):
        trial_type = f"{prefix}pc{pc_idx}" if prefix else f"pc{pc_idx}"

        for i, tr_idx in enumerate(valid_indices):
            # Onset is relative to the start of the run image (not absolute)
            onset_time = tr_idx * tr_duration

            modulated_events.append({
                'trial_type': trial_type,
                'onset':      onset_time,
                'duration':   tr_duration,
                'modulation': pc_data[i, pc_idx],
            })

    events_df = pd.DataFrame(modulated_events)

    logger.debug(
        "Created modulated events | prefix='%s' | components=%d | "
        "valid_trs=%d | total_rows=%d",
        prefix, n_components, len(valid_indices), len(events_df)
    )

    return events_df


def apply_hrf_convolution(
    img,
    events_df: pd.DataFrame,
    valid_indices: np.ndarray,
    tr_duration: float = 2.0,
) -> tuple[np.ndarray, list]:
    """
    Fit a GLM and return HRF-convolved features for the valid TRs.

    Uses nilearn's FirstLevelModel with the SPM + derivative + dispersion
    HRF model and AR(1) noise. Each PC regressor is convolved with the HRF,
    capturing the delayed haemodynamic response to the neural signal
    represented by that component.

    Only the rows corresponding to valid_indices are returned â€” the rest
    of the design matrix (TRs with no linguistic content) is discarded.

    Parameters
    ----------
    img : Nifti1Image
        The fMRI run image. Required by FirstLevelModel to determine
        the number of timepoints and construct the design matrix.
    events_df : pd.DataFrame
        Modulated events from create_modulated_events().
        Columns: trial_type, onset, duration, modulation.
    valid_indices : np.ndarray
        TR indices to extract from the full design matrix.
    tr_duration : float
        TR length in seconds (default 2.0).

    Returns
    -------
    convolved_features : np.ndarray, shape (n_valid_trs, n_pc_columns)
        HRF-convolved PC regressors at the valid TR timepoints.
    pc_columns : list of str
        Column names of the PC regressors in the design matrix.
        Used by the caller to separate current vs future components.
    """
    logger.debug(
        "Fitting FirstLevelModel | events=%d | unique trial types=%d",
        len(events_df), events_df['trial_type'].nunique()
    )

    # Log modulation statistics for the first 3 trial types
    for trial_type in events_df['trial_type'].unique()[:3]:
        mods = events_df[events_df['trial_type'] == trial_type]['modulation'].values
        logger.debug(
            "  %s modulation: mean=%.4f std=%.4f range=[%.4f, %.4f]",
            trial_type, np.mean(mods), np.std(mods), np.min(mods), np.max(mods)
        )

    # Fit the GLM â€” this performs the HRF convolution
    first_level_model = FirstLevelModel(
        t_r=tr_duration,
        hrf_model='spm + derivative + dispersion',
        standardize=False,
        noise_model='ar1',
    )
    first_level_model.fit(img, events_df)
    design_matrix = first_level_model.design_matrices_[0]

    logger.debug("Design matrix shape: %s", design_matrix.shape)

    # Extract only the PC columns (excludes the intercept and drift terms
    # that nilearn adds automatically)
    pc_columns = [col for col in design_matrix.columns if 'pc' in col]

    logger.debug(
        "PC columns found: %d | first 5: %s",
        len(pc_columns), pc_columns[:5]
    )

    if len(pc_columns) == 0:
        raise ValueError(
            "No PC columns found in the design matrix. "
            "Check that create_modulated_events() used 'pc' in the trial_type names."
        )

    # Extract only the valid TR rows from the design matrix
    convolved_features = design_matrix[pc_columns].values[valid_indices]

    logger.info(
        "HRF convolution done | convolved features shape: %s", convolved_features.shape
    )

    return convolved_features, pc_columns