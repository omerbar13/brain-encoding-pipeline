def get_run_words(
    feature_data,
    run_start_time,
    run_end_time
):
    """
    Extract all words belonging to a run.

    Returns
    -------
    list of tuples:
        (onset, word, duration)
    """

    run_words = [
        (onset, data['speech'], data['duration'])
        for onset, data in feature_data.items()
        if run_start_time <= onset < run_end_time
    ]

    return sorted(run_words, key=lambda x: x[0])