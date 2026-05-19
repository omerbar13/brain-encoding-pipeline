# src/brain/roi.py

import logging
import numpy as np
from nilearn import datasets, image
from nilearn.image import load_img, new_img_like, resample_to_img, concat_imgs

logger = logging.getLogger(__name__)


def load_atlas(n_rois: int = 400, yeo_networks: int = 17, resolution_mm: int = 2):
    """
    Fetch the Schaefer 2018 parcellation atlas.

    This is separated from build_roi_mask() so the atlas is only
    downloaded/loaded once per pipeline run, not once per ROI.

    Parameters
    ----------
    n_rois : int
        Number of parcels (100, 200, 300, 400, 500, 600, 800, 1000).
    yeo_networks : int
        Yeo network resolution (7 or 17).
    resolution_mm : int
        Atlas resolution in mm (1 or 2).

    Returns
    -------
    atlas_img : Nifti1Image
        The parcellation image — each voxel value is a parcel index.
    labels : list of str
        Parcel label names corresponding to each index.
    """
    logger.info(
        "Fetching Schaefer atlas: n_rois=%d, yeo_networks=%d, resolution=%dmm",
        n_rois, yeo_networks, resolution_mm
    )
    schaefer = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois,
        yeo_networks=yeo_networks,
        resolution_mm=resolution_mm
    )
    atlas_img = image.load_img(schaefer['maps'])
    labels = schaefer['labels']

    logger.info("Atlas loaded: %d parcels", n_rois)
    return atlas_img, labels


def build_roi_mask(run_img, atlas_img, roi_indices: list[int]) -> np.ndarray:
    """
    Build a boolean 3-D voxel mask for a set of ROI parcel indices.

    The atlas is resampled to the voxel space of run_img using nearest-
    neighbour interpolation (required for a parcellation — linear
    interpolation would create meaningless fractional parcel values).

    Parameters
    ----------
    run_img : Nifti1Image
        A single fMRI run image, used only for its affine / voxel grid.
    atlas_img : Nifti1Image
        The parcellation atlas returned by load_atlas().
    roi_indices : list of int
        Parcel indices to include in the mask (e.g. [149, 150, ..., 194]).

    Returns
    -------
    roi_mask : np.ndarray, shape (x, y, z), dtype bool
        True wherever a voxel belongs to one of the requested parcels.
    """
    atlas_resampled = resample_to_img(atlas_img, run_img, interpolation='nearest')
    atlas_data = atlas_resampled.get_fdata()

    roi_mask = np.zeros_like(atlas_data, dtype=bool)
    for roi_idx in roi_indices:
        roi_mask = roi_mask | (atlas_data == roi_idx)

    logger.debug(
        "ROI mask built: %d voxels selected from %d parcels",
        np.sum(roi_mask), len(roi_indices)
    )
    return roi_mask


def process_runs(
    training_runs: list,
    test_run,
    roi_indices: list[int],
    atlas_img,
) -> tuple:
    """
    Apply ROI masking to training runs and the test run.

    For training runs:
      - Loads each NIfTI image
      - Builds the ROI mask in that run's voxel space
      - Zeros out all non-ROI voxels
      - Concatenates all masked training images along the time axis

    For the test run:
      - Applies the same masking procedure
      - Returns the mask separately so downstream code can use it
        for voxel-wise variance filtering

    Parameters
    ----------
    training_runs : list of Nifti1Image or str
        Training run images (or paths). Run 4 should already be excluded
        by load_all_runs() before calling this function.
    test_run : Nifti1Image or str
        The held-out test run image (or path).
    roi_indices : list of int
        Parcel indices defining the ROI (from config.ROI_CONFIGS).
    atlas_img : Nifti1Image
        Pre-loaded atlas from load_atlas(). Passed in so the atlas is
        not re-fetched on every call.

    Returns
    -------
    concat_img : Nifti1Image
        All masked training runs concatenated along the time axis.
        Shape: (x, y, z, total_training_timepoints)
    test_masked_img : Nifti1Image
        The masked test run image.
        Shape: (x, y, z, test_timepoints)
    roi_mask : np.ndarray, shape (x, y, z), dtype bool
        The ROI mask in the test run's voxel space.
        (Used downstream for voxel-wise variance filtering.)
    """
    logger.info(
        "Processing %d training run(s) + 1 test run for %d ROI parcels",
        len(training_runs), len(roi_indices)
    )

    # --- Mask and collect training runs ---
    training_imgs = []

    for i, run in enumerate(training_runs):
        run_img = load_img(run)
        roi_mask = build_roi_mask(run_img, atlas_img, roi_indices)

        run_data = run_img.get_fdata()
        # roi_mask is 3-D; add a time axis so it broadcasts across all TRs
        masked_data = run_data * roi_mask[..., np.newaxis]
        masked_img = new_img_like(run_img, masked_data)
        training_imgs.append(masked_img)

        logger.info(
            "Training run %d: shape=%s, ROI voxels=%d",
            i + 1, run_img.shape, np.sum(roi_mask)
        )

    concat_img = concat_imgs(training_imgs)

    # --- Mask the test run ---
    test_img = load_img(test_run)
    roi_mask = build_roi_mask(test_img, atlas_img, roi_indices)

    test_data = test_img.get_fdata()
    test_masked_data = test_data * roi_mask[..., np.newaxis]
    test_masked_img = new_img_like(test_img, test_masked_data)

    logger.info("ROI mask coverage : %d voxels", np.sum(roi_mask))
    logger.info("Training data shape: %s", concat_img.shape)
    logger.info("Test data shape    : %s", test_masked_img.shape)

    return concat_img, test_masked_img, roi_mask