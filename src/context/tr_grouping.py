def group_words_by_tr(
    run_words,
    run_start_time,
    tr_duration=2.0
):
    """
    Group run words into TR bins.

    Returns
    -------
    words_by_tr : dict
        TR index -> list of (onset, word, duration)
    """

    words_by_tr = {}

    for onset, word, duration in run_words:

        tr_idx = int((onset - run_start_time) / tr_duration)

        if tr_idx not in words_by_tr:
            words_by_tr[tr_idx] = []

        words_by_tr[tr_idx].append(
            (onset, word, duration)
        )

    return words_by_tr


def find_last_words_per_tr(words_by_tr):
    """
    Select the final word occurring inside each TR.

    Returns
    -------
    dict
        TR index -> (onset, word, duration)
    """

    last_words_by_tr = {}

    for tr_idx, words in words_by_tr.items():

        last_word = max(
            words,
            key=lambda x: x[0]
        )

        last_words_by_tr[tr_idx] = last_word

    return last_words_by_tr