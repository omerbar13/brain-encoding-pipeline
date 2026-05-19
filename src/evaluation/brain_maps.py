# src/evaluation/brain_maps.py

import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from nilearn import datasets, plotting, surface
from nilearn.image import new_img_like

logger = logging.getLogger(__name__)


def create_brain_map(
    voxel_scores: np.ndarray,
    data_img,
    output_dir: str,
) -> None:
    """
    Project voxel-wise scores onto the fsaverage cortical surface and save
    one PNG per metric (correlation, combined score, prediction gain).

    Three maps are produced:
    - correlation_scores  : current-word-only brain score  (column 3, diverging)
    - r2_scores           : combined brain score           (column 4, sequential)
    - prediction_gains    : gain = combined - current      (column 5, diverging)

    Each map is rendered from 4 views: lateral left, medial left,
    lateral right, medial right — saved as a single wide figure.

    Parameters
    ----------
    voxel_scores : np.ndarray, shape (n_voxels, 6)
        Output of process_voxels(). Columns: [x, y, z, current, combined, gain].
    data_img : Nifti1Image
        Any run image in the same voxel space — used only for its affine
        to construct the output NIfTI volume map.
    output_dir : str
        Directory where PNG files are saved.
    """
    if len(voxel_scores) == 0:
        logger.error("No voxel scores available — skipping brain map creation.")
        return

    os.makedirs(output_dir, exist_ok=True)

    original_shape = data_img.shape[:3]

    metrics = {
        'correlation_scores': (3, 'Correlation Map'),
        'r2_scores':          (4, 'R² Map'),
        'prediction_gains':   (5, 'Prediction Gain Map'),
    }

    # Sequential colormap for the combined-score map
    custom_cmap = LinearSegmentedColormap.from_list(
        'custom', ['blue', 'green', 'yellow', 'orange', 'red']
    )

    fsaverage = datasets.fetch_surf_fsaverage()

    for metric_name, (score_idx, title) in metrics.items():

        # --- Build 3-D volume map ---
        volume_map = np.zeros(original_shape)
        for score in voxel_scores:
            if not np.isnan(score[score_idx]):
                x, y, z = int(score[0]), int(score[1]), int(score[2])
                volume_map[x, y, z] = score[score_idx]

        map_img = new_img_like(data_img, volume_map)

        # --- Compute colour scale ---
        non_zero = volume_map[volume_map != 0]
        if len(non_zero) > 0:
            if metric_name in ('correlation_scores', 'prediction_gains'):
                abs_max = max(abs(np.percentile(non_zero, [1, 99])))
                vmin, vmax = -abs_max, abs_max
            else:
                vmin = 0
                vmax = np.percentile(non_zero, 99)
        else:
            vmin, vmax = (-1, 1) if metric_name in ('correlation_scores', 'prediction_gains') else (0, 1)

        # --- Project volume → surface ---
        texture_left  = surface.vol_to_surf(map_img, fsaverage.pial_left,  interpolation='linear')
        texture_right = surface.vol_to_surf(map_img, fsaverage.pial_right, interpolation='linear')

        # --- Plot 4-view figure ---
        fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'projection': '3d'})
        fig.suptitle(title, fontsize=12, x=0.45)

        view_hemi_pairs = [
            ('lateral', 'left'), ('medial', 'left'),
            ('lateral', 'right'), ('medial', 'right'),
        ]

        use_diverging = metric_name in ('correlation_scores', 'prediction_gains')

        for i, (view, hemi) in enumerate(view_hemi_pairs):
            texture  = texture_left  if hemi == 'left'  else texture_right
            infl     = fsaverage.infl_left  if hemi == 'left'  else fsaverage.infl_right
            sulc     = fsaverage.sulc_left  if hemi == 'left'  else fsaverage.sulc_right
            cmap     = 'RdBu_r' if use_diverging else custom_cmap

            plotting.plot_surf_stat_map(
                infl, texture,
                hemi=hemi, view=view,
                colorbar=False,
                bg_map=sulc,
                cmap=cmap,
                axes=axes[i],
                vmin=vmin, vmax=vmax,
                bg_on_data=True,
            )

        # --- Colorbar ---
        fig.subplots_adjust(top=1, right=0.85, wspace=0, hspace=0.00002, left=0.05, bottom=0)
        cbar_ax = fig.add_axes([0.95, 0.05, 0.02, 0.8])

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm   = plt.cm.ScalarMappable(cmap='RdBu_r' if use_diverging else custom_cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label(title, rotation=270, labelpad=15)

        out_path = os.path.join(output_dir, f'{metric_name}_surface.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved brain map: %s", out_path)

    logger.info("Brain maps saved to %s", output_dir)