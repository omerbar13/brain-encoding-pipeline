from src.context.run_words import get_run_words

def create_word_prediction_window(feature_data, word_embeddings, run_length, run_durations, run_index,
                                 window_sizes={'short': 3, 'medium': 6, 'long': 9}, max_gap_seconds=2.0):
    """
    Creates word-level prediction windows of multiple sizes based on the LAST word in each TR

    Steps:
    1) Retrieves runtime for test run
    2) Retrieves words, onsets, and duration for words within the test run
    3) Maps words to TR
    4) Groups words to TR
    5) Finds the last word within each TR
    6) For each size, creates dictionary of word-future-pairs through their onset
    7) For each size, finds future words following exclusionary rules
    8) Creates embeddings for all window sizes
    """
    print(f"\nCreating word-level prediction windows (multiple sizes) for run {run_index + 1}...")

    # 1. Determine runtime for the run
    run_start_time = sum(run_durations[:run_index])
    run_end_time = run_start_time + run_durations[run_index]
    tr_duration = 2.0
    print(f"Run timing: start={run_start_time:.2f}s, end={run_end_time:.2f}s, duration={run_durations[run_index]:.2f}s")

    # LOGGING1 information about feature_data
    print(f"Feature data contains {len(feature_data)} total entries")
    sample_keys = list(feature_data.keys())[:5]  # Get 5 sample keys
    print("Feature data structure (5 sample entries):")
    for key in sample_keys:
        print(f"  Onset {key:.2f}s: '{feature_data[key]['speech']}', duration: {feature_data[key]['duration']:.2f}s")

    # 2. Retrieve all words for current run
    run_words = get_run_words(
        feature_data,
        run_start_time,
        run_end_time
    )

    print(f"Found {len(run_words)} words in run {run_index + 1}")

    # LOGGING2 Check embedding stats
    n_words_with_embeddings = sum(1 for onset, _, _ in run_words if onset in word_embeddings)
    print(f"Words with embeddings: {n_words_with_embeddings}/{len(run_words)} "
          f"({n_words_with_embeddings / len(run_words) * 100:.1f}%)")

    # 3. Maps each word within run_words to its TR
    word_to_tr = {onset: int((onset - run_start_time) / tr_duration)
                  for onset, _, _ in run_words}

    # 4. Group words by TR
    words_by_tr = {}
    for onset, word, duration in run_words:
        tr_idx = word_to_tr[onset]
        if tr_idx not in words_by_tr:
            words_by_tr[tr_idx] = []
        words_by_tr[tr_idx].append((onset, word, duration))

    # 5. Find the last word in each TR (highest onset time)
    last_words_by_tr = {}
    for tr_idx, words in words_by_tr.items():
        last_word = max(words, key=lambda x: x[0])
        last_words_by_tr[tr_idx] = last_word

    print(f"Found {len(last_words_by_tr)} TRs with words")
    print(f"Using only the last word in each TR for future window creation")

    # LOGGING3 Enhanced logging for debugging
    print("\nDetailed word grouping by TR (first 3 TRs with words):")
    sample_tr_indices = list(words_by_tr.keys())[:3]
    for tr_idx in sample_tr_indices:
        print(f"\n  TR {tr_idx} contains {len(words_by_tr[tr_idx])} words:")
        for i, (onset, word, duration) in enumerate(words_by_tr[tr_idx]):
            print(f"    Word {i+1}: '{word}' at {onset:.2f}s (duration: {duration:.2f}s)")

    # LOGGING4 Sample of last words per TR for debugging
    sample_trs = list(last_words_by_tr.keys())[:5] if last_words_by_tr else []
    print("\nSample of last words per TR:")
    for tr_idx in sample_trs:
        onset, word, duration = last_words_by_tr[tr_idx]
        print(f"  TR {tr_idx}: '{word}' at {onset:.2f}s (duration: {duration:.2f}s)")

    # LOGGING5 Check if these are actually the last words
    print("\nVerification of last word selection (first 3 TRs):")
    for tr_idx in sample_tr_indices:
        if tr_idx in last_words_by_tr:
            last_onset, last_word, _ = last_words_by_tr[tr_idx]
            all_onsets = [onset for onset, _, _ in words_by_tr[tr_idx]]
            max_onset = max(all_onsets)
            is_correct = last_onset == max_onset
            print(f"  TR {tr_idx}: Selected '{last_word}' at {last_onset:.2f}s | "
                  f"Max onset is {max_onset:.2f}s | Correct: {is_correct}")

    # 6. Initialize dictionaries for each window size
    word_future_pairs = {size_name: {} for size_name in window_sizes}

    # 7. Sort the list again to ensure chronological order
    sorted_run_words = sorted(run_words, key=lambda x: x[0])

    # 8. MAIN PART: WINDOW CREATION - For each window size, create future windows
    for size_name, max_window_size in window_sizes.items():
        print(f"\nCreating {size_name} windows (size: {max_window_size} words)...")

        # For each last word in a TR, create a future window of this size
        for tr_idx, (current_onset, current_word, current_duration) in last_words_by_tr.items():
            # 8.1 Find position of current word in sorted list
            current_idx = next((i for i, (onset, _, _) in enumerate(sorted_run_words)
                               if onset == current_onset), -1)

            if current_idx == -1 or current_idx >= len(sorted_run_words) - 1:
                continue  # Skip if we can't find word or it's the last word

            # 8.2 Initialize list and end of words to measure the 2 seconds exclusion rule
            future_onsets = []
            current_end = current_onset + current_duration

            # 8.3 Enact inclusionary criteria
            j = current_idx + 1  # Start from the next word after our current one
            while len(future_onsets) < max_window_size and j < len(sorted_run_words):
                next_onset, _, next_duration = sorted_run_words[j]
                time_gap = next_onset - current_end

                # Break if gap is too large and we have at least 2 words
                if time_gap > max_gap_seconds and len(future_onsets) >= 2:
                    break

                future_onsets.append(next_onset)
                current_end = next_onset + next_duration
                j += 1

            # 8.4 Only keep windows with at least 2 future words
            if len(future_onsets) >= 2:
                word_future_pairs[size_name][current_onset] = future_onsets

        # Log results for this window size
        print(f"Created {len(word_future_pairs[size_name])} {size_name} windows")

    # LOGGING6 Print sample future windows for each size
    print("\nSample future windows for each size:")
    for size_name, pairs in word_future_pairs.items():
        if not pairs:
            print(f"  No {size_name} windows created")
            continue

        sample_size = min(3, len(pairs))
        sample_onsets = list(pairs.keys())[:sample_size]

        print(f"\n{size_name.upper()} WINDOWS (size: {window_sizes[size_name]} words):")
        for onset in sample_onsets:
            sample_word = next((word for o, word, _ in run_words if o == onset), "Unknown")
            sample_duration = next((duration for o, _, duration in run_words if o == onset), 0)
            future_onsets = pairs[onset]

            print(f"  Base word: '{sample_word}' at {onset:.2f}s (duration: {sample_duration:.2f}s)")
            print(f"  Future words: {len(future_onsets)}")

            word_end = onset + sample_duration
            for j, future_onset in enumerate(future_onsets):
                future_word = next((word for o, word, _ in run_words if o == future_onset), "Unknown")
                future_duration = next((duration for o, _, duration in run_words if o == future_onset), 0)
                time_gap = future_onset - word_end

                print(f"    {j+1}: '{future_word}' at {future_onset:.2f}s, gap: {time_gap:.2f}s")
                word_end = future_onset + future_duration

    # 9. Get embedding dimension
    embedding_dim = next(iter(word_embeddings.values())).shape[0] if word_embeddings else 768
    print(f"Embedding dimension: {embedding_dim}")

    # 10. Create output dictionary to store results for all window sizes
    results = {}

    # 11. Process each window size
    for size_name, max_window_size in window_sizes.items():
        print(f"\nProcessing embeddings for {size_name} windows...")

        # Size-specific window dimensions
        future_window_size = max_window_size * embedding_dim
        final_future_embeddings = np.zeros((run_length, future_window_size))
        valid_mask = np.zeros(run_length, dtype=bool)

        # 11.1 For each TR with a last word, retrieve and process future embeddings
        for tr_idx, (current_onset, _, _) in last_words_by_tr.items():
            # 11.2 Skip if this word doesn't have a future window for this size
            if current_onset not in word_future_pairs[size_name]:
                continue

            # 11.3 Skip if TR is out of bounds
            if tr_idx >= run_length:
                continue

            # 11.4 Get future embeddings for this window
            future_onsets = word_future_pairs[size_name][current_onset]
            future_embeddings_list = []

            for onset in future_onsets:
                if onset in word_embeddings:
                    future_embeddings_list.append(word_embeddings[onset])
                else:
                    future_embeddings_list.append(np.zeros(embedding_dim))

            # 11.5 Pad/truncate as needed
            while len(future_embeddings_list) < max_window_size:
                future_embeddings_list.append(np.zeros(embedding_dim))
            future_embeddings_list = future_embeddings_list[:max_window_size]

            # 11.6 Stack and flatten while preserving information
            stacked = np.stack(future_embeddings_list)
            flattened = stacked.flatten()

            # 11.7 Store in final output
            final_future_embeddings[tr_idx] = flattened
            valid_mask[tr_idx] = True

        # Log information about this window size
        trs_with_windows = np.sum(valid_mask)
        print(f"{size_name.capitalize()} windows summary:")
        print(f"  TRs with valid windows: {trs_with_windows} out of {run_length}")
        print(f"  Coverage: {trs_with_windows / run_length * 100:.1f}% of TRs")
        print(f"  Embedding dimensionality: {future_window_size}")

        # Store results for this window size
        results[size_name] = (final_future_embeddings, valid_mask)

    return results