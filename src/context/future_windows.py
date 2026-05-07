def create_future_windows(
    sorted_run_words,
    last_words_by_tr,
    max_window_size,
    max_gap_seconds=2.0
):
    """
    Create future prediction windows for each TR.

    Parameters
    ----------
    sorted_run_words : list
        Chronologically ordered words:
        (onset, word, duration)

    last_words_by_tr : dict
        TR -> final word in TR

    max_window_size : int
        Maximum number of future words

    max_gap_seconds : float
        Maximum allowed temporal discontinuity

    Returns
    -------
    dict
        current_onset -> list of future onsets
    """

    word_future_pairs = {}

    for tr_idx, (
        current_onset,
        current_word,
        current_duration
    ) in last_words_by_tr.items():

        # Find current word index
        current_idx = next(
            (
                i for i, (onset, _, _) in enumerate(sorted_run_words)
                if onset == current_onset
            ),
            -1
        )

        if current_idx == -1:
            continue

        if current_idx >= len(sorted_run_words) - 1:
            continue

        future_onsets = []

        current_end = (
            current_onset +
            current_duration
        )

        j = current_idx + 1

        while (
            len(future_onsets) < max_window_size
            and j < len(sorted_run_words)
        ):

            next_onset, _, next_duration = sorted_run_words[j]

            time_gap = next_onset - current_end

            if (
                time_gap > max_gap_seconds
                and len(future_onsets) >= 2
            ):
                break

            future_onsets.append(next_onset)

            current_end = (
                next_onset +
                next_duration
            )

            j += 1

        if len(future_onsets) >= 2:
            word_future_pairs[current_onset] = future_onsets

    return word_future_pairs