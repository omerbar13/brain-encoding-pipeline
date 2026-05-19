# src/pipeline/run_pipeline.py

import logging
import os
import datetime
import numpy as np
import pandas as pd

from src.pipeline.config import (
    DATA_PATH, TSV_PATH, BASE_OUTPUT_DIR,
    TR, N_COMPONENTS, ALPHA,
    N_ROIS, YEO_NETWORKS, ATLAS_RESOLUTION,
    ROI_CONFIGS, WINDOW_SIZES,
)
from src.data.annotations    import process_annotations
from src.data.load_runs      import load_all_runs
from src.data.run_durations  import get_run_durations_with_logging
from src.features.embeddings import get_word_embeddings
from src.alignment.temporal_alignment import align_embeddings_to_tr
from src.alignment.tr_features        import average_embeddings_per_tr
from src.context.prediction_windows   import create_word_prediction_window
from src.brain.roi                    import load_atlas, build_roi_mask, process_runs
from src.models.ridge                 import process_voxels
from src.evaluation.brain_maps        import create_brain_map
from src.evaluation.reports           import analyze_timing_effects, create_summary_report
from src.evaluation.timing            import create_timing_visualization

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def run_pipeline():

    # ------------------------------------------------------------------
    # Output directory (timestamped, matching notebook convention)
    # ------------------------------------------------------------------
    timestamp  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(BASE_OUTPUT_DIR, f'analysis_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # ------------------------------------------------------------------
    # 1. Run durations
    # ------------------------------------------------------------------
    logger.info("Step 1: Loading run durations...")
    run_durations = get_run_durations_with_logging(DATA_PATH, tr_duration=TR)

    # ------------------------------------------------------------------
    # 2. Annotations
    # ------------------------------------------------------------------
    logger.info("Step 2: Processing annotations...")
    feature_data, events_df, speech_df = process_annotations(TSV_PATH)

    # ------------------------------------------------------------------
    # 3. Load fMRI runs (skip run 4)
    # ------------------------------------------------------------------
    logger.info("Step 3: Loading fMRI runs...")
    all_runs = load_all_runs(DATA_PATH, num_runs=8, skip_runs=(4,))

    # ------------------------------------------------------------------
    # 4. Generate word embeddings (once for all runs)
    # ------------------------------------------------------------------
    logger.info("Step 4: Generating word embeddings...")
    word_embeddings = get_word_embeddings(feature_data, context_length=50)

    # ------------------------------------------------------------------
    # 5. Load atlas (once)
    # ------------------------------------------------------------------
    logger.info("Step 5: Loading Schaefer atlas...")
    atlas_img, labels = load_atlas(
        n_rois=N_ROIS,
        yeo_networks=YEO_NETWORKS,
        resolution_mm=ATLAS_RESOLUTION,
    )

    # ------------------------------------------------------------------
    # 6. Per-run embedding alignment and future window construction
    # ------------------------------------------------------------------
    logger.info("Step 6: Aligning embeddings and building context windows...")
    all_run_embeddings    = []
    all_run_masks         = []
    all_run_future_windows = []

    for run_idx, run_img in enumerate(all_runs):
        actual_run_num = run_idx + 1 if run_idx < 3 else run_idx + 2
        logger.info("--- Processing run %d (actual run %d) ---", run_idx + 1, actual_run_num)

        # Cumulative start time (accounting for skipped run 4)
        run_start_time = 0.0
        for i in range(actual_run_num):
            if i == 3:
                run_start_time += run_durations[i]
            elif i < actual_run_num - 1:
                run_start_time += run_durations[i]

        run_length = run_img.shape[-1]

        tr_embeddings = align_embeddings_to_tr(
            word_embeddings, run_start_time, run_length, tr_duration=TR
        )
        current_embeddings, current_mask = average_embeddings_per_tr(
            tr_embeddings, run_length
        )
        future_windows_dict = create_word_prediction_window(
            feature_data=feature_data,
            word_embeddings=word_embeddings,
            run_length=run_length,
            run_durations=run_durations,
            run_index=actual_run_num - 1,
            window_sizes=WINDOW_SIZES,
            max_gap_seconds=2.0,
        )

        all_run_embeddings.append(current_embeddings)
        all_run_masks.append(current_mask)
        all_run_future_windows.append(future_windows_dict)

    # ------------------------------------------------------------------
    # 7. Train / test split (last run held out)
    # ------------------------------------------------------------------
    train_runs       = all_runs[:-1]
    test_run         = all_runs[-1]
    train_embeddings = all_run_embeddings[:-1]
    test_embeddings  = all_run_embeddings[-1]
    train_masks      = all_run_masks[:-1]
    test_mask        = all_run_masks[-1]

    # ------------------------------------------------------------------
    # 8. Outer loop: ROI × window size
    # ------------------------------------------------------------------
    all_results = {}

    for roi_name, roi_indices in ROI_CONFIGS.items():
        logger.info("===== ROI: %s =====", roi_name)
        roi_output_dir = os.path.join(output_dir, roi_name)
        os.makedirs(roi_output_dir, exist_ok=True)

        roi_mask = build_roi_mask(all_runs[-1], atlas_img, roi_indices)
        all_results[roi_name] = {}

        for window_name, window_size in WINDOW_SIZES.items():
            logger.info("--- Window: %s (%d words) ---", window_name, window_size)
            window_output_dir = os.path.join(roi_output_dir, window_name)
            os.makedirs(window_output_dir, exist_ok=True)

            # Extract embeddings for this window size
            train_future_embs  = [r[window_name][0] for r in all_run_future_windows[:-1]]
            train_future_masks = [r[window_name][1] for r in all_run_future_windows[:-1]]
            test_future_embs   = all_run_future_windows[-1][window_name][0]
            test_future_mask   = all_run_future_windows[-1][window_name][1]

            # Voxel-wise encoding model
            voxel_scores = process_voxels(
                test_img=test_run,
                train_imgs=train_runs,
                test_current_embeddings=test_embeddings,
                test_current_mask=test_mask,
                test_future_embeddings=test_future_embs,
                test_future_mask=test_future_mask,
                train_current_embeddings_list=train_embeddings,
                train_current_mask_list=train_masks,
                train_future_embeddings_list=train_future_embs,
                train_future_mask_list=train_future_masks,
                events_df=events_df,
                roi_mask=roi_mask,
                alpha=ALPHA,
                run_durations=run_durations,
                n_components=N_COMPONENTS,
                tr_duration=TR,
            )

            # Save results
            np.save(
                os.path.join(window_output_dir, f'{roi_name}_{window_name}_voxel_scores.npy'),
                voxel_scores
            )

            # Metadata file
            valid_scores = voxel_scores[~np.isnan(voxel_scores).any(axis=1)]
            with open(os.path.join(window_output_dir, 'analysis_metadata.txt'), 'w') as f:
                f.write(f"ROI: {roi_name}\n")
                f.write(f"Window: {window_name} ({window_size} words)\n")
                f.write(f"Total voxels: {len(voxel_scores)}\n")
                f.write(f"Valid voxels: {len(valid_scores)}\n")
                f.write(f"Alpha: {ALPHA}\n")
                f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if len(valid_scores) > 0:
                    f.write(f"\nMean brain score:      {np.mean(valid_scores[:, 3]):.4f}\n")
                    f.write(f"Mean combined score:   {np.mean(valid_scores[:, 4]):.4f}\n")
                    f.write(f"Mean prediction gain:  {np.mean(valid_scores[:, 5]):.4f}\n")
                    f.write(
                        f"Positive gain voxels:  "
                        f"{np.sum(valid_scores[:, 5] > 0) / len(valid_scores) * 100:.1f}%\n"
                    )

            # Evaluation outputs
            timing_analysis = analyze_timing_effects(voxel_scores, feature_data)
            create_brain_map(voxel_scores, test_run, window_output_dir)
            create_summary_report(window_output_dir, voxel_scores, timing_analysis)
            create_timing_visualization(window_output_dir, feature_data, voxel_scores)

            # Store summary for final comparison table
            if len(valid_scores) > 0:
                all_results[roi_name][window_name] = {
                    'brain_score':       float(np.mean(valid_scores[:, 3])),
                    'final_brain_score': float(np.mean(valid_scores[:, 4])),
                    'prediction_score':  float(np.mean(valid_scores[:, 5])),
                    'valid_voxels':      len(valid_scores),
                    'positive_gain_pct': float(
                        np.sum(valid_scores[:, 5] > 0) / len(valid_scores) * 100
                    ),
                }

    # ------------------------------------------------------------------
    # 9. Final comparison table
    # ------------------------------------------------------------------
    logger.info("===== FINAL COMPARISON =====")
    rows = []
    for roi_name, windows in all_results.items():
        for window_name, result in windows.items():
            rows.append({
                'ROI':               roi_name,
                'Window':            window_name,
                'Window Size':       WINDOW_SIZES[window_name],
                'Brain Score':       result['brain_score'],
                'Final Brain Score': result['final_brain_score'],
                'Prediction Score':  result['prediction_score'],
                'Valid Voxels':      result['valid_voxels'],
                'Positive Gain %':   result['positive_gain_pct'],
            })

    if rows:
        comparison_df = pd.DataFrame(rows)
        comparison_df['Rank'] = comparison_df['Prediction Score'].rank(ascending=False)
        logger.info("\n%s", comparison_df.to_string(index=False))

        best = comparison_df.loc[comparison_df['Prediction Score'].idxmax()]
        logger.info(
            "Best configuration: ROI=%s | Window=%s | Prediction Score=%.4f",
            best['ROI'], best['Window'], best['Prediction Score']
        )

        comparison_df.to_csv(os.path.join(output_dir, 'comparison_table.csv'), index=False)

    logger.info("Pipeline complete. Results saved to: %s", output_dir)


if __name__ == "__main__":
    run_pipeline()