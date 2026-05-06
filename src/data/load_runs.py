import os
from nilearn.image import load_img


def load_all_runs(base_dir, num_runs=8, skip_runs=(4,)):
    """
    Load fMRI run images from a directory, skipping specified runs.

    Parameters
    ----------
    base_dir : str
        Directory containing the hyperaligned NIfTI run files.
    num_runs : int
        Total number of runs expected.
    skip_runs : tuple
        Runs to skip (e.g., run 4).

    Returns
    -------
    list
        List of loaded NIfTI images.
    """

    runs = []

    for run_num in range(1, num_runs + 1):

        if run_num in skip_runs:
            print(f"Skipping run {run_num}")
            continue

        run_path = os.path.join(
            base_dir,
            f"group_avg_movie_run_{run_num}_hyperaligned.nii"
        )

        try:
            run_img = load_img(run_path)
            runs.append(run_img)
            print(f"Loaded run {run_num}")

        except Exception as e:
            print(f"Error loading run {run_num}: {run_path}")
            raise e

    print(f"\nTotal runs loaded: {len(runs)}")
    return runs