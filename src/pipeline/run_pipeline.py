# run_pipeline.py

from src.context.prediction_windows import create_word_prediction_window
from src.context.prediction_windows import create_word_prediction_window
from src.data.load_runs import load_all_runs
from src.data.run_durations import get_run_durations_with_logging
from src.features.embeddings import get_word_embeddings
from src.data.load_runs import load_all_runs
from src.data.run_durations import get_run_durations_with_logging
from src.features.embeddings import get_word_embeddings

def run_pipeline():

    print("\n===== STARTING FULL BRAIN PIPELINE =====")

    # 1. Load MRI runs
    runs = load_all_runs(data_path)

    # 2. Get run durations
    run_durations = get_run_durations_with_logging(data_path)

    # 3. Load annotations
    feature_data, events_df, speech_df = process_annotations(tsv_path)

    # 4. Generate embeddings
    word_embeddings = get_word_embeddings(feature_data)
    
    # 5. Build current + future embeddings for test run
    window_outputs = create_word_prediction_window(
        feature_data=feature_data,
        word_embeddings=word_embeddings,
        run_length=len(runs[-1]),   # or your TR count if different
        run_durations=run_durations,
        run_index=len(runs) - 1
    )

    test_current_embeddings = window_outputs["current_embeddings"]
    test_current_mask = window_outputs["current_mask"]

    test_future_embeddings_short = window_outputs["future"]["short"]["future_embeddings"]
    test_future_mask_short = window_outputs["future"]["short"]["future_mask"]

    # 5. (example placeholder) build train/test split
    train_imgs = runs[:-1]
    test_img = runs[-1]

    # 6. Align embeddings per run
    # (this will use your existing logic)

    # 7. Build masks + embeddings (CURRENT + FUTURE)
    # test_current_embeddings, test_future_embeddings, etc.

    # 8. Run voxel model
    voxel_scores = process_voxels_ba_revised(
        test_img=test_img,
        train_imgs=train_imgs,
        test_current_embeddings=...,
        test_current_mask=...,
        test_future_embeddings=...,
        test_future_mask=...,
        train_current_embeddings_list=...,
        train_current_mask_list=...,
        train_future_embeddings_list=...,
        train_future_mask_list=...,
        events_df=events_df,
        ba_mask=roi_mask,
        alpha=1000,
        run_durations=run_durations
    )

    # 9. Analysis outputs
    timing = analyze_timing_effects(voxel_scores, feature_data)

    create_summary_report(base_results_dir, voxel_scores, timing)
    create_brain_map(voxel_scores, test_img, base_results_dir)
    create_timing_visualization(base_results_dir, feature_data, voxel_scores)

    print("\n===== PIPELINE COMPLETE =====")


if __name__ == "__main__":
    run_pipeline()