import numpy as np


def align_embeddings_to_tr(
    word_embeddings,
    run_start_time,
    run_length,
    tr_duration=2.0
):
    """
    Align word embeddings to fMRI TRs using temporal binning.

    Parameters
    ----------
    word_embeddings : dict
        onset (float) -> embedding vector
    run_start_time : float
        start time of the current run
    run_length : int
        number of TRs in the run
    tr_duration : float
        duration of a single TR (seconds)

    Returns
    -------
    dict
        TR index -> list of embeddings
    """

    tr_embeddings = {tr_idx: [] for tr_idx in range(run_length)}

    run_end_time = run_start_time + (run_length * tr_duration)

    for onset, embedding in word_embeddings.items():

        if run_start_time <= onset < run_end_time:

            tr_idx = int((onset - run_start_time) / tr_duration)

            if 0 <= tr_idx < run_length:
                tr_embeddings[tr_idx].append(embedding)

    return tr_embeddings