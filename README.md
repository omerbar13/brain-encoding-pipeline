# Neural Encoding Pipeline: EEG/fMRI + NLP Transformer Alignment

A modular machine learning pipeline for aligning transformer-based language representations with fMRI brain activity to study neural encoding of speech and language processing.


## Key Features

- Transformer-based NLP embeddings for language representation
- fMRI-based neural encoding using ROI-level brain parcellation (Schaefer atlas)
- Cross-validated regression models (Ridge, PCA-based pipelines)
- Temporal analysis using multiple window sizes
- Reproducible experimental pipeline with modular architecture


## Tech Stack

- Python
- PyTorch / Transformers
- scikit-learn
- Nilearn (fMRI analysis)
- NumPy, Pandas
- Matplotlib


## Pipeline Overview

1. fMRI data loading and preprocessing (StudyForrest dataset)
2. Brain parcellation using Schaefer atlas (400 ROIs)
3. Text processing using transformer-based language models
4. Alignment of linguistic embeddings with neural signals
5. Encoding model training (Ridge regression + cross-validation)
6. Evaluation across brain regions and time windows


## Repository Structure

src/               Core pipeline modules  
notebooks/         Experimental notebooks  
configs/           Experiment configurations  
scripts/           Execution scripts  
results/           Outputs and figures  


## Data

This project uses the StudyForrest fMRI dataset.  
Due to data usage restrictions, raw neuroimaging data is not included in this repository.

Instructions for dataset access and preprocessing are provided in the documentation.


## Author Contribution

This project was developed as part of a collaborative research internship at the Donders Institute for Brain, Cognition and Behaviour.

Core contributions include:
- Machine learning pipeline design and implementation
- NLP-fMRI alignment modeling
- Encoding model development and evaluation workflows



