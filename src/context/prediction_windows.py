from src.context.run_words import get_run_words
from src.context.tr_grouping import (
    group_words_by_tr,
    find_last_words_per_tr
)
from src.context.future_windows import create_future_windows
from src.context.window_embeddings import (
    build_future_embedding_matrix
)
import numpy as np

def create_current_embeddings(last_words_by_tr, word_embeddings, run_length):
    embedding_dim = next(iter(word_embeddings.values())).shape[0]

    current_embeddings = np.zeros((run_length, embedding_dim))
    valid_mask = np.zeros(run_length, dtype=bool)

    for tr_idx, (onset, _, _) in last_words_by_tr.items():

        if onset in word_embeddings and tr_idx < run_length:
            current_embeddings[tr_idx] = word_embeddings[onset]
            valid_mask[tr_idx] = True

    return current_embeddings, valid_mask

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

    words_by_tr = group_words_by_tr(
        run_words,
        run_start_time,
        tr_duration
    )

    last_words_by_tr = find_last_words_per_tr(
        words_by_tr
    )

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



    ## new
    # 6. Initialize dictionaries for each window size
    word_future_pairs = {}

    # 7. Ensure chronological ordering
    sorted_run_words = sorted(
        run_words,
        key=lambda x: x[0]
    )

    # 8. Create future windows
    for size_name, max_window_size in window_sizes.items():
        
        print(
            f"\nCreating {size_name} windows "
            f"(size: {max_window_size} words)..."
        )

        word_future_pairs[size_name] = create_future_windows(
            sorted_run_words=sorted_run_words,
            last_words_by_tr=last_words_by_tr,
            max_window_size=max_window_size,
            max_gap_seconds=max_gap_seconds
        )

        print(
            f"Created "
            f"{len(word_future_pairs[size_name])} "
            f"{size_name} windows"
        )


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

        final_future_embeddings, valid_mask = (
            build_future_embedding_matrix(
                last_words_by_tr=last_words_by_tr,
                word_future_pairs=word_future_pairs[size_name],
                word_embeddings=word_embeddings,
                run_length=run_length,
                max_window_size=max_window_size,
                embedding_dim=embedding_dim
            )
        )       

        # Log information about this window size
        trs_with_windows = np.sum(valid_mask)
        print(f"{size_name.capitalize()} windows summary:")
        print(f"  TRs with valid windows: {trs_with_windows} out of {run_length}")
        print(f"  Coverage: {trs_with_windows / run_length * 100:.1f}% of TRs")
        print(f"  Embedding dimensionality: {final_future_embeddings.shape[1]}")

        # Store results for this window size
        results[size_name] = {
            "future_embeddings": final_future_embeddings,
            "future_mask": valid_mask
        }

    current_embeddings, current_mask = create_current_embeddings(
        last_words_by_tr,
        word_embeddings,
        run_length
    )

    return {
        "current_embeddings": current_embeddings,
        "current_mask": current_mask,
        "future": results
    }