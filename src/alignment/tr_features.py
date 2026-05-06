import numpy as np


def average_embeddings_per_tr(tr_embeddings, run_length):
    """
    Aggregate word embeddings into TR-level representations for encoding models.

    This function:
    - Averages embeddings per TR
    - Preserves temporal alignment
    - Creates a validity mask for regression

    Parameters
    ----------
    tr_embeddings : dict
        TR index -> list of embeddings
    run_length : int
        Total number of TRs in the run

    Returns
    -------
    current_embeddings : np.ndarray
        Shape: (run_length, embedding_dim)
    valid_mask : np.ndarray
        Boolean array indicating TRs with valid data
    """

    print("\nAveraging embeddings per TR...")

    # --------------------------------------------------
    # 1. Detect embedding dimension
    # --------------------------------------------------
    embedding_dim = None

    for embeddings in tr_embeddings.values():
        if len(embeddings) > 0:
            embedding_dim = embeddings[0].shape[0]
            break

    if embedding_dim is None:
        raise ValueError("No valid embeddings found in any TR")

    print(f"Embedding dimension: {embedding_dim}")
    print(f"Total TRs: {run_length}")

    # --------------------------------------------------
    # 2. Initialize outputs
    # --------------------------------------------------
    current_embeddings = np.zeros((run_length, embedding_dim))
    valid_mask = np.zeros(run_length, dtype=bool)

    # --------------------------------------------------
    # 3. Fill TR-level representations
    # --------------------------------------------------
    for tr_idx, embeddings in tr_embeddings.items():

        if tr_idx >= run_length:
            continue

        if len(embeddings) > 0:
            current_embeddings[tr_idx] = np.mean(embeddings, axis=0)
            valid_mask[tr_idx] = True

    # --------------------------------------------------
    # 4. Diagnostics
    # --------------------------------------------------
    trs_with_words = np.sum(valid_mask)

    print(
        f"TRs with words: {trs_with_words}/{run_length} "
        f"({trs_with_words / run_length * 100:.1f}%)"
    )

    return current_embeddings, valid_mask