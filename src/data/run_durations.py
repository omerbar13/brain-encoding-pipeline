import os
import numpy as np


def get_run_durations_with_logging(data_path, log=True):
    """
    Load or compute durations for each fMRI run.

    Parameters
    ----------
    data_path : str
        Path to dataset directory
    log : bool
        Whether to print debug information

    Returns
    -------
    list
        Run durations in seconds
    """


    structure_file = os.path.join(data_path, "structure.csv")

    if log:
        print(f"Loading run durations from: {structure_file}")

    
    import pandas as pd
    df = pd.read_csv(structure_file)

    # IMPORTANT: adjust column name based on your file
    if "duration" in df.columns:
        run_durations = df["duration"].tolist()
    else:
        raise ValueError("No 'duration' column found in structure.csv")

    if log:
        print("Run durations loaded successfully")
        print(run_durations)

    return run_durations