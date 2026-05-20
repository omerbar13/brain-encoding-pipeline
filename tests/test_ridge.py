# tests/test_ridge.py

import numpy as np
import pytest
from src.models.ridge import standardize_timeseries, compute_correlation
from sklearn.linear_model import Ridge


def test_standardize_normal_signal():
    """Output should have mean â‰ˆ 0 and std â‰ˆ 1."""
    ts = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = standardize_timeseries(ts)
    assert abs(np.mean(result)) < 1e-10
    assert abs(np.std(result) - 1.0) < 1e-6


def test_standardize_flat_signal_returns_zeros():
    """A flat (zero-variance) signal should return a zero vector, not raise."""
    ts = np.ones(10) * 5.0
    result = standardize_timeseries(ts)
    np.testing.assert_array_equal(result, np.zeros(10))


def test_compute_correlation_perfect():
    """A model that predicts perfectly should return correlation â‰ˆ 1."""
    X = np.arange(20).reshape(-1, 1).astype(float)
    y = np.arange(20).astype(float)
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    r = compute_correlation(model, X, y)
    assert r > 0.99


def test_compute_correlation_constant_prediction_returns_zero():
    """If predictions are constant, correlation is undefined â€” should return 0."""
    X = np.ones((10, 1))
    y = np.arange(10).astype(float)
    model = Ridge(alpha=1e10)   # extreme regularisation â†’ near-zero coefficients
    model.fit(X, y)
    r = compute_correlation(model, X, y)
    assert r == 0.0 or abs(r) < 0.01