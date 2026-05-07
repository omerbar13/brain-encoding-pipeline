import numpy as np


def build_future_embedding_matrix(
    last_words_by_tr,
    word_future_pairs,
    word_embeddings,
    run_length,
    max_window_size,
    embedding_dim
):
    """
    Build flattened future-context embedding matrices per TR.

    Returns
    -------
    final_future_embeddings : np.ndarray
    valid_mask : np.ndarray
    """

    future_window_size = max_window_size * embedding_dim

    final_future_embeddings = np.zeros(
        (run_length, future_window_size)
    )

    valid_mask = np.zeros(
        run_length,
        dtype=bool
    )

    for tr_idx, (current_onset, _, _) in last_words_by_tr.items():

        if current_onset not in word_future_pairs:
            continue

        if tr_idx >= run_length:
            continue

        future_onsets = word_future_pairs[current_onset]

        future_embeddings_list = []

        for onset in future_onsets:

            if onset in word_embeddings:
                future_embeddings_list.append(
                    word_embeddings[onset]
                )
            else:
                future_embeddings_list.append(
                    np.zeros(embedding_dim)
                )

        while len(future_embeddings_list) < max_window_size:
            future_embeddings_list.append(
                np.zeros(embedding_dim)
            )

        future_embeddings_list = future_embeddings_list[:max_window_size]

        stacked = np.stack(future_embeddings_list)

        flattened = stacked.flatten()

        final_future_embeddings[tr_idx] = flattened

        valid_mask[tr_idx] = True

    return final_future_embeddings, valid_mask