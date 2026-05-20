# tests/test_alignment.py

import numpy as np
import pytest
from src.alignment.temporal_alignment import align_embeddings_to_tr
from src.alignment.tr_features import average_embeddings_per_tr


def test_align_embeddings_basic():
    """Words within a run's time window are assigned to the correct TR."""
    embs = {
        1.0: np.ones(10),        # onset 1.0s → TR 0 (0–2s)
        3.5: np.ones(10) * 2,    # onset 3.5s → TR 1 (2–4s)
        6.1: np.ones(10) * 3,    # onset 6.1s → TR 3 (6–8s)
    }
    result = align_embeddings_to_tr(embs, run_start_time=0.0, run_length=5, tr_duration=2.0)

    assert 0 in result and len(result[0]) == 1
    assert 1 in result and len(result[1]) == 1
    assert 3 in result and len(result[3]) == 1
    assert len(result[2]) == 0   # TR 2 has no words


def test_align_embeddings_outside_window_ignored():
    """Words outside the run's time window are not assigned to any TR."""
    embs = {
        99.0: np.ones(10),   # well outside a 5-TR run starting at 0
    }
    result = align_embeddings_to_tr(embs, run_start_time=0.0, run_length=5, tr_duration=2.0)
    assert all(len(v) == 0 for v in result.values())


def test_average_embeddings_valid_mask():
    """TRs with embeddings get a non-zero vector; empty TRs get zeros and False mask."""
    tr_embeddings = {
        0: [np.ones(4)],
        1: [np.ones(4) * 2, np.ones(4) * 4],   # two words → should average to 3
        2: [],
    }
    embeddings, mask = average_embeddings_per_tr(tr_embeddings, run_length=3)

    assert embeddings.shape == (3, 4)
    assert mask[0] == True
    assert mask[1] == True
    assert mask[2] == False
    np.testing.assert_array_almost_equal(embeddings[1], np.ones(4) * 3)
    np.testing.assert_array_equal(embeddings[2], np.zeros(4))