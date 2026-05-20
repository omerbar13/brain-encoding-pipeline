# Brain Encoding Pipeline

A modular Python implementation of a voxel-wise brain encoding pipeline. The project maps GPT-2 language representations onto fMRI brain activity recorded while participants listened to a narrated story, and asks a specific question: does adding *future* linguistic context improve how well a model can predict brain responses, beyond what the current word alone explains?

The codebase is a full refactor of a research notebook into a structured, testable Python project.

---

## Background

Brain encoding models work by taking a stimulus representation — here, a word embedding from GPT-2 — and training a linear model to predict the fMRI response in each voxel. The better the model's predictions correlate with the actual brain signal, the more that representation is said to "encode" activity in that region.

This pipeline extends the standard approach by constructing two models per voxel: one using only the current word's embedding, and one that also includes a window of future words. The difference in predictive accuracy between the two (the *prediction gain*) is the main outcome of interest. The motivation comes from work by Caucheteux et al. on predictive processing in language-selective brain regions.

The fMRI data comes from the [StudyForrest](https://www.studyforrest.org/) dataset — a public neuroimaging dataset where participants listened to an audio description of the film *Forrest Gump*.

---

## Pipeline Overview

```
Raw fMRI runs (.nii)          Speech annotations (.tsv)
        │                               │
        ▼                               ▼
 load_all_runs()              process_annotations()
        │                               │
        │                               ▼
        │                    get_word_embeddings()
        │                     (GPT-2, context=50)
        │                               │
        │                               ▼
        │                  align_embeddings_to_tr()
        │                  average_embeddings_per_tr()
        │                               │
        │                               ▼
        │                create_word_prediction_window()
        │                  (short / medium / long)
        │                               │
        └───────────────────────────────┘
                          │
                          ▼
                   build_roi_mask()
               (Schaefer 2018 atlas)
                          │
                          ▼
                  process_voxels()
             ┌────────────────────────┐
             │  PCA (fit on train)    │
             │  HRF convolution       │
             │  Ridge regression      │
             │  current model  ──┐    │
             │  combined model ──┤    │
             └───────────────────┼────┘
                                 │
                          prediction gain
                                 │
                    ┌────────────┴─────────────┐
                    ▼                          ▼
            create_brain_map()     create_summary_report()
            (fsaverage surface)    (Wilcoxon test, stats)
```

---

## Project Structure

```
brain-encoding-pipeline/
├── configs/                        # Experiment configuration files
├── notebooks/                      # Original research notebook
├── results/                        # Pipeline outputs (gitignored)
├── scripts/                        # Utility scripts
├── src/
│   ├── alignment/
│   │   ├── temporal_alignment.py   # Align word onsets to TR indices
│   │   └── tr_features.py          # Average embeddings per TR, build validity mask
│   ├── brain/
│   │   └── roi.py                  # Atlas loading, ROI masking, run processing
│   ├── context/
│   │   ├── future_windows.py       # Future word window construction
│   │   ├── prediction_windows.py   # Per-run window embedding matrices
│   │   ├── run_words.py            # Word-to-run assignment
│   │   ├── tr_grouping.py          # Group words by TR
│   │   └── window_embeddings.py    # Aggregate embeddings across windows
│   ├── data/
│   │   ├── annotations.py          # Load and filter speech annotation TSV
│   │   ├── load_runs.py            # Load NIfTI fMRI runs
│   │   └── run_durations.py        # Compute per-run durations from NIfTI shapes
│   ├── evaluation/
│   │   ├── brain_maps.py           # Project voxel scores to fsaverage surface
│   │   ├── reports.py              # Summary statistics and Wilcoxon test
│   │   └── timing.py               # Inter-word interval vs gain scatter plot
│   ├── features/
│   │   └── embeddings.py           # GPT-2 contextual word embeddings
│   ├── models/
│   │   ├── hrf.py                  # HRF convolution via nilearn FirstLevelModel
│   │   └── ridge.py                # Voxel-wise Ridge regression encoding model
│   └── pipeline/
│       ├── config.py               # All constants and paths in one place
│       └── run_pipeline.py         # End-to-end pipeline entry point
└── tests/
    ├── test_alignment.py
    ├── test_annotations.py
    ├── test_ridge.py
    └── test_roi.py
```

---

## Setup

Python 3.10 or higher is required.

Clone the repo and install:

```bash
git clone https://github.com/omerbar13/brain-encoding-pipeline.git
cd brain-encoding-pipeline
pip install -e .
```

---

## Configuration

All paths and parameters live in `src/pipeline/config.py`. Before running, either edit that file directly or set environment variables:

```bash
export BRAIN_DATA_PATH=/path/to/fmri/runs
export BRAIN_TSV_PATH=/path/to/fg_rscut_ad_ger_speech_tagged.tsv
export BRAIN_OUTPUT_DIR=./results
```

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TR` | `2.0` | Repetition time in seconds |
| `N_COMPONENTS` | `20` | PCA components per embedding type |
| `ALPHA` | `1000.0` | Ridge regression regularisation |
| `WINDOW_SIZES` | `{short:3, medium:6, long:9}` | Future context window sizes (words) |

---

## Running the Pipeline

```bash
python -m src.pipeline.run_pipeline
```

The pipeline will create a timestamped output directory under `results/` containing, for each ROI and window size combination:

- `voxel_scores.npy` — raw results array, shape `(n_voxels, 6)`
- `analysis_summary.txt` — Wilcoxon test and score statistics
- `*_surface.png` — brain maps projected onto fsaverage
- `timing_analysis.png` — inter-word interval vs prediction gain
- `analysis_metadata.txt` — run parameters and coverage stats
- `comparison_table.csv` — ranked summary across all configurations

---

## Data

Raw fMRI data is not included in this repository. The pipeline was developed using the [StudyForrest](https://www.studyforrest.org/) dataset, specifically the hyperaligned group-average movie-watching runs. Data access is available through the StudyForrest project — see their site for details.

Expected inputs:

- `group_avg_movie_run_{1..8}_hyperaligned.nii` — one file per run
- `fg_rscut_ad_ger_speech_tagged.tsv` — word-level speech annotations with onset, duration, POS tags

Note: run 4 is excluded from model training and testing (its duration is still used for timing calculations).

---

## Tests

The test suite covers core logic without requiring any real fMRI data — all tests use synthetic arrays:

```bash
python -m pytest tests/ -v
```

```
tests/test_alignment.py::test_align_embeddings_basic PASSED
tests/test_alignment.py::test_align_embeddings_outside_window_ignored PASSED
tests/test_alignment.py::test_average_embeddings_valid_mask PASSED
tests/test_annotations.py::test_filters_phoneme_and_sentence PASSED
tests/test_annotations.py::test_filters_narrator PASSED
tests/test_annotations.py::test_time_interval_computed PASSED
tests/test_annotations.py::test_missing_column_raises PASSED
tests/test_ridge.py::test_standardize_normal_signal PASSED
tests/test_ridge.py::test_standardize_flat_signal_returns_zeros PASSED
tests/test_ridge.py::test_compute_correlation_perfect PASSED
tests/test_ridge.py::test_compute_correlation_constant_prediction_returns_zero PASSED
tests/test_roi.py::test_build_roi_mask_correct_voxels PASSED
tests/test_roi.py::test_build_roi_mask_multiple_indices PASSED
tests/test_roi.py::test_build_roi_mask_empty_indices PASSED
14 passed
```

---

## Key Design Decisions

**No data leakage in PCA.** PCA is fit on training runs only and applied to the test run using the same fitted components. Fitting on all data together would allow test-run variance to influence the principal components.

**Run 4 excluded but timed correctly.** Run 4 is never loaded or used in training/testing, but its duration (976 s) is still included in cumulative timing calculations so that word onsets and TR indices stay correctly aligned across all other runs.

**HRF model choice.** The SPM + derivative + dispersion HRF model with AR(1) noise is used throughout, matching the original experimental setup. Changing this would affect the convolved feature matrices and therefore the regression results.

**Prediction gain as the outcome.** Rather than reporting a single brain score, the pipeline produces a *gain* — the improvement in voxel-wise correlation when future context is added to the current-word model. This isolates the contribution of predictive linguistic processing from the baseline language response.

---

## Author

Omer Bar
