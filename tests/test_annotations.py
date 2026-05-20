# tests/test_annotations.py

import pytest
import pandas as pd
import tempfile
import os
from src.data.annotations import process_annotations


def _write_tsv(rows: list[dict], path: str):
    pd.DataFrame(rows).to_csv(path, sep='\t', index=False)


def test_filters_phoneme_and_sentence():
    """PHONEME and SENTENCE rows are removed; real words are kept."""
    rows = [
        {'onset': 1.0, 'duration': 0.3, 'speaker': 'S1', 'text': 'hello', 'pos': 'NN'},
        {'onset': 1.3, 'duration': 0.1, 'speaker': 'S1', 'text': 'h',     'pos': 'PHONEME'},
        {'onset': 2.0, 'duration': 0.5, 'speaker': 'S1', 'text': 'world', 'pos': 'NN'},
        {'onset': 2.5, 'duration': 0.8, 'speaker': 'S1', 'text': None,    'pos': 'SENTENCE'},
    ]
    with tempfile.NamedTemporaryFile(suffix='.tsv', delete=False, mode='w') as f:
        path = f.name
    try:
        _write_tsv(rows, path)
        feature_data, events_df, speech_df = process_annotations(path)
        assert len(speech_df) == 2
        assert set(speech_df['text']) == {'hello', 'world'}
    finally:
        os.unlink(path)


def test_filters_narrator():
    """Rows where the third column contains NARRATOR are removed."""
    rows = [
        {'onset': 1.0, 'duration': 0.3, 'speaker': 'NARRATOR', 'text': 'once', 'pos': 'NN'},
        {'onset': 2.0, 'duration': 0.4, 'speaker': 'S1',        'text': 'upon', 'pos': 'NN'},
    ]
    with tempfile.NamedTemporaryFile(suffix='.tsv', delete=False, mode='w') as f:
        path = f.name
    try:
        _write_tsv(rows, path)
        feature_data, events_df, speech_df = process_annotations(path)
        assert len(speech_df) == 1
        assert list(speech_df['text']) == ['upon']
    finally:
        os.unlink(path)


def test_time_interval_computed():
    """time_interval = next_onset - (onset + duration), NaN for last word."""
    rows = [
        {'onset': 1.0, 'duration': 0.3, 'speaker': 'S1', 'text': 'hello', 'pos': 'NN'},
        {'onset': 2.0, 'duration': 0.4, 'speaker': 'S1', 'text': 'world', 'pos': 'NN'},
    ]
    with tempfile.NamedTemporaryFile(suffix='.tsv', delete=False, mode='w') as f:
        path = f.name
    try:
        _write_tsv(rows, path)
        _, _, speech_df = process_annotations(path)
        # interval for 'hello': 2.0 - (1.0 + 0.3) = 0.7
        assert abs(speech_df.iloc[0]['time_interval'] - 0.7) < 1e-6
        assert pd.isna(speech_df.iloc[1]['time_interval'])
    finally:
        os.unlink(path)


def test_missing_column_raises():
    """A TSV missing a required column raises ValueError with a clear message."""
    rows = [{'onset': 1.0, 'duration': 0.3, 'text': 'hello'}]  # 'pos' missing
    with tempfile.NamedTemporaryFile(suffix='.tsv', delete=False, mode='w') as f:
        path = f.name
    try:
        _write_tsv(rows, path)
        with pytest.raises(ValueError, match="pos"):
            process_annotations(path)
    finally:
        os.unlink(path)