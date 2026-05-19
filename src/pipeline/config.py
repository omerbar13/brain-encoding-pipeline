# src/pipeline/config.py

import os

# --- Data paths ---
# Set these via environment variables or edit directly before running.
DATA_PATH       = os.environ.get("BRAIN_DATA_PATH",   "/path/to/fmri/runs")
TSV_PATH        = os.environ.get("BRAIN_TSV_PATH",    "/path/to/annotations.tsv")
BASE_OUTPUT_DIR = os.environ.get("BRAIN_OUTPUT_DIR",  "./results")

# --- Imaging parameters ---
TR           = 2.0   # Repetition time in seconds
N_COMPONENTS = 20    # PCA components (following Caucheteux et al.)
ALPHA        = 1000.0  # Ridge regression regularisation strength

# --- Atlas parameters ---
N_ROIS           = 400
YEO_NETWORKS     = 17
ATLAS_RESOLUTION = 2  # mm

# --- ROI definitions (Schaefer 2018 parcel indices) ---
ROI_CONFIGS = {
    "DMN":         list(range(149, 195)) + list(range(358, 391)),
    "association": list(range(195, 201)) + list(range(391, 400)),
    "attention":   list(range(60,  108)) + list(range(259, 313)),
}

# --- Context window sizes (words) ---
WINDOW_SIZES = {
    "short":  3,
    "medium": 6,
    "long":   9,
}