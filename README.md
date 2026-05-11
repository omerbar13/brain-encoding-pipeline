# Brain Encoding Pipeline

A modular research codebase for aligning transformer-based language representations with fMRI data in a brain encoding pipeline.

This project refactors an exploratory research notebook into reusable Python modules for data loading, language embedding generation, temporal alignment, context-window construction, and pipeline orchestration.

## Project Overview

The pipeline combines natural language processing and neuroimaging analysis to study how language representations relate to brain activity during movie-watching and speech comprehension.

Core components include:

- GPT-2 based contextual word embeddings
- Temporal alignment of word embeddings to fMRI TRs
- Current-word and future-context representation construction
- Multi-window context modeling (`short`, `medium`, `long`)
- Modular structure for future voxel-wise encoding analysis

## Repository Structure

```text
src/
├── alignment/      # Align word embeddings to fMRI TRs
├── context/        # Build current/future linguistic context windows
├── data/           # Load fMRI runs and run-duration metadata
├── features/       # Generate transformer-based word embeddings
├── pipeline/       # Lightweight pipeline entrypoint and configuration
├── models/         # Reserved for voxel-wise encoding models
├── evaluation/     # Reserved for reports and evaluation utilities
└── brain/          # Reserved for brain-map and ROI utilities
```

Additional folders:

```text
configs/            # Experiment configuration files
notebooks/          # Exploratory notebooks
results/            # Local output files and generated results
scripts/            # Utility scripts
```

## Data

Raw neuroimaging data is not included in this repository.

The original experiments used fMRI data and annotation files that are subject to dataset and institutional usage restrictions. Users should obtain the relevant data through the appropriate official channels and configure paths locally.

Expected local inputs include:

- Hyperaligned fMRI run files (`.nii`)
- Speech annotation file (`.tsv`)
- Brain mask and atlas files where applicable

## Current Status

This repository is a cleaned modular refactor of a research pipeline originally developed in notebook form.

The `src/` modules contain reusable components for the main pipeline stages, while the full end-to-end experimental workflow can be extended from the provided pipeline scaffold.

The current focus of this repository is code organization, modularity, and reproducibility rather than redistribution of restricted data or notebook-specific intermediate outputs.

## Main Components

### Data Loading

`src/data/`

Contains utilities for loading fMRI runs and run-duration information.

### Embedding Generation

`src/features/`

Contains GPT-2 based contextual word embedding generation.

### Temporal Alignment

`src/alignment/`

Contains functions for aligning word-level embeddings to fMRI TRs and averaging embeddings per TR.

### Context Windows

`src/context/`

Contains logic for constructing future word prediction windows and converting them into TR-aligned embedding matrices.

### Pipeline Entry Point

`src/pipeline/run_pipeline.py`

Provides a lightweight entrypoint showing how the modular components connect.

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

The current pipeline scaffold can be run from the project root with:

```bash
python -m src.pipeline.run_pipeline
```

Before running full experiments, update local paths in:

```text
src/pipeline/config.py
```

## Notes

This repository contains research code intended to illustrate the structure, methodology, and implementation of the brain encoding project.

The code is provided as a portfolio and research demonstration. It is not a standalone distribution of the full experimental dataset or institutional research environment.

The repository does not include raw fMRI data, subject-level data, or restricted experimental files.

## Author

Omer Bar