# src/data/annotations.py

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def process_annotations(tsv_path: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Load and filter the speech annotation TSV file.

    Filters applied (in order):
    - Removes rows where pos == 'PHONEME'
    - Removes rows where text is null/NaN
    - Removes rows where pos == 'SENTENCE'
    - Removes rows where the third column contains 'NARRATOR'

    Also computes an inter-word time_interval column:
        time_interval = onset_of_next_word - (onset + duration)

    Parameters
    ----------
    tsv_path : str
        Path to the speech annotation TSV file
        (e.g. fg_rscut_ad_ger_speech_tagged.tsv)

    Returns
    -------
    feature_data : dict
        Mapping of onset (float) -> {
            'speech'         : str,   the word text
            'duration'       : float, word duration in seconds
            'original_onset' : float, same as the key
            'time_interval'  : float, gap to the next word (NaN for last word)
        }
    events_df : pd.DataFrame
        nilearn FirstLevelModel-compatible events table.
        Columns: trial_type, onset, duration
    speech_df : pd.DataFrame
        Filtered DataFrame with all original columns plus time_interval.
    """

    logger.info("Processing annotations file: %s", tsv_path)

    try:
        df = pd.read_csv(tsv_path, sep='\t')
        logger.info("Loaded raw TSV: %d rows, %d columns", df.shape[0], df.shape[1])
        logger.debug("Columns: %s", ', '.join(df.columns))

        # --- Validate required columns before doing anything else ---
        required_cols = ['onset', 'duration', 'pos', 'text']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Required column '{col}' not found in TSV. "
                    f"Found columns: {df.columns.tolist()}"
                )

        # --- Apply filters ---
        # .copy() ensures the filters only affect speech_df, not the original df
        speech_df = df.copy()
        speech_df = speech_df[speech_df['pos'] != 'PHONEME']
        speech_df = speech_df[speech_df['text'].notna()]
        speech_df = speech_df[speech_df['pos'] != 'SENTENCE']

        # The third column (index 2) holds speaker tags — remove NARRATOR lines
        speech_df = speech_df[
            ~speech_df[speech_df.columns[2]].str.contains('NARRATOR', na=False)
        ]

        # --- Compute inter-word time intervals ---
        # shift(-1) aligns the next row's onset with the current row,
        # so: interval = next_onset - (current_onset + current_duration)
        speech_df['time_interval'] = (
            speech_df['onset'].shift(-1)
            - (speech_df['onset'] + speech_df['duration'])
        )

        # --- Build outputs ---

        # nilearn-compatible events table
        events_df = pd.DataFrame({
            'trial_type': ['speech'] * len(speech_df),
            'onset':      speech_df['onset'].values,
            'duration':   speech_df['duration'].values,
        })

        # Main data structure used by the rest of the pipeline.
        # Keys are onset times; values are per-word metadata dicts.
        feature_data = {
            row['onset']: {
                'speech':          row['text'],
                'duration':        row['duration'],
                'original_onset':  row['onset'],
                'time_interval':   row['time_interval'],
            }
            for _, row in speech_df.iterrows()
        }

        # --- Logging / diagnostics ---
        logger.info(
            "Filtered speech_df: %d rows | feature_data: %d entries | events_df: %d rows",
            len(speech_df), len(feature_data), len(events_df),
        )

        logger.debug("First 25 rows of speech_df:\n%s", speech_df.head(25).to_string())
        logger.debug("Last 25 rows of speech_df:\n%s",  speech_df.tail(25).to_string())

        logger.debug(
            "First 25 entries of feature_data:\n%s",
            pd.DataFrame(list(feature_data.values())[:25]).to_string()
        )
        logger.debug(
            "Last 25 entries of feature_data:\n%s",
            pd.DataFrame(list(feature_data.values())[-25:]).to_string()
        )

    except Exception as e:
        logger.error("Failed to process annotations from %s: %s", tsv_path, e)
        raise

    return feature_data, events_df, speech_df