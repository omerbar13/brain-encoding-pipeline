# tests/test_roi.py

import numpy as np
import pytest
import nibabel as nib
from src.brain.roi import build_roi_mask


def _make_fake_atlas(shape=(10, 10, 10)):
    """Create a minimal NIfTI image where each voxel value is its flat index mod 5."""
    data = np.arange(np.prod(shape)).reshape(shape) % 5
    return nib.Nifti1Image(data.astype(np.float32), affine=np.eye(4))


def test_build_roi_mask_correct_voxels():
    """Mask should be True exactly where atlas value == requested parcel index."""
    atlas_img = _make_fake_atlas()
    run_img   = _make_fake_atlas()   # same space â€” no resampling needed

    mask = build_roi_mask(run_img, atlas_img, roi_indices=[1])
    atlas_data = atlas_img.get_fdata()

    np.testing.assert_array_equal(mask, atlas_data == 1)


def test_build_roi_mask_multiple_indices():
    """Multiple parcel indices should be OR-ed together in the mask."""
    atlas_img = _make_fake_atlas()
    run_img   = _make_fake_atlas()

    mask = build_roi_mask(run_img, atlas_img, roi_indices=[1, 2])
    atlas_data = atlas_img.get_fdata()

    expected = (atlas_data == 1) | (atlas_data == 2)
    np.testing.assert_array_equal(mask, expected)


def test_build_roi_mask_empty_indices():
    """Empty roi_indices list should return an all-False mask."""
    atlas_img = _make_fake_atlas()
    run_img   = _make_fake_atlas()

    mask = build_roi_mask(run_img, atlas_img, roi_indices=[])
    assert mask.dtype == bool
    assert not np.any(mask)