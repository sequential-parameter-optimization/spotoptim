# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the _apply_penalty_NA method with y_history parameter."""

import numpy as np
from spotoptim import SpotOptim


def test_apply_penalty_with_history():
    """Test that penalty is computed from y_history, not from y itself."""
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1), bounds=[(-5, 5), (-5, 5)], verbose=False
    )

    # Historical values with known statistics
    y_history = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # max=5.0, mean=3.0, std=1.414 (ddof=1)
    # Expected penalty: 5.0 + 3*1.414 = 9.24

    # New values with NaN/inf
    y_new = np.array([6.0, np.nan, np.inf, 7.0])

    y_clean = opt._apply_penalty_NA(y_new, y_history=y_history)

    # Check all values are finite
    assert np.all(np.isfinite(y_clean))

    # Check non-NaN values are unchanged
    assert y_clean[0] == 6.0
    assert y_clean[3] == 7.0

    # Check NaN/inf were replaced with penalty around 9.24
    # Should be > max(y_history) = 5.0
    assert y_clean[1] > 5.0
    assert y_clean[2] > 5.0

    # Should be approximately max + 3*std (with small random noise)
    expected_penalty = 5.0 + 3.0 * np.std(y_history, ddof=1)
    assert abs(y_clean[1] - expected_penalty) < 1.0  # Allow for random noise
    assert abs(y_clean[2] - expected_penalty) < 1.0


def test_apply_penalty_without_history():
    """Test fallback behavior when y_history is None."""
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1), bounds=[(-5, 5), (-5, 5)], verbose=False
    )

    # No history provided, should use y itself
    y_new = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

    y_clean = opt._apply_penalty_NA(y_new, y_history=None)

    # Check all values are finite
    assert np.all(np.isfinite(y_clean))

    # Check non-NaN values are unchanged
    assert y_clean[0] == 1.0
    assert y_clean[1] == 2.0
    assert y_clean[3] == 4.0
    assert y_clean[4] == 5.0

    # NaN should be replaced with penalty computed from finite values in y_new
    # max(1,2,4,5)=5, std([1,2,4,5])≈1.826
    assert y_clean[2] > 5.0


def test_apply_penalty_history_all_finite():
    """Test when y_history has all finite values but y has NaN/inf."""
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1), bounds=[(-5, 5), (-5, 5)], verbose=False
    )

    y_history = np.array([10.0, 20.0, 30.0, 40.0])
    y_new = np.array([np.nan, np.inf, -np.inf])

    y_clean = opt._apply_penalty_NA(y_new, y_history=y_history)

    assert np.all(np.isfinite(y_clean))

    # All should be replaced with penalty based on y_history
    # max=40, std≈12.909
    expected_penalty = 40.0 + 3.0 * np.std(y_history, ddof=1)
    for val in y_clean:
        assert val > 40.0
        assert abs(val - expected_penalty) < 2.0


def test_apply_penalty_history_with_nan():
    """Test when y_history itself contains NaN/inf values."""
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1), bounds=[(-5, 5), (-5, 5)], verbose=False
    )

    # History with some NaN/inf
    y_history = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
    y_new = np.array([np.nan, 7.0])

    y_clean = opt._apply_penalty_NA(y_new, y_history=y_history)

    assert np.all(np.isfinite(y_clean))

    # Should use only finite values from history: [1, 3, 5]
    # max=5, std≈2.0
    assert y_clean[0] > 5.0
    assert y_clean[1] == 7.0


def test_apply_penalty_explicit_penalty_value():
    """Test with explicit penalty_value parameter."""
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1), bounds=[(-5, 5), (-5, 5)], verbose=False
    )

    y_history = np.array([1.0, 2.0, 3.0])
    y_new = np.array([np.nan, 4.0, np.inf])

    # Use explicit penalty value of 100
    y_clean = opt._apply_penalty_NA(y_new, y_history=y_history, penalty_value=100.0)

    assert np.all(np.isfinite(y_clean))

    # NaN/inf should be replaced with ~100 (plus small noise)
    assert 99.0 < y_clean[0] < 101.0
    assert y_clean[1] == 4.0
    assert 99.0 < y_clean[2] < 101.0


def test_apply_penalty_insufficient_history():
    """Test when history has only one finite value."""
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1),
        bounds=[(-5, 5), (-5, 5)],
        penalty=500.0,  # Set explicit penalty
        verbose=False,
    )

    y_history = np.array([42.0])  # Only one finite value
    y_new = np.array([np.nan, np.inf])

    y_clean = opt._apply_penalty_NA(y_new, y_history=y_history)

    assert np.all(np.isfinite(y_clean))

    # Should use self.penalty = 500.0 as fallback
    assert y_clean[0] > 400.0
    assert y_clean[1] > 400.0


def test_apply_penalty_empty_history():
    """Test when history is empty but y has finite values."""
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1), bounds=[(-5, 5), (-5, 5)], verbose=False
    )

    y_history = np.array([])  # Empty history
    y_new = np.array([1.0, np.nan, 3.0])

    y_clean = opt._apply_penalty_NA(y_new, y_history=y_history)

    assert np.all(np.isfinite(y_clean))

    # Should fall back to using y itself
    assert y_clean[0] == 1.0
    assert y_clean[1] > 3.0  # Should use max(1,3) + 3*std
    assert y_clean[2] == 3.0


def test_apply_penalty_all_nan_in_y_and_history():
    """Test when both y and y_history are all NaN/inf."""
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1), bounds=[(-5, 5), (-5, 5)], verbose=False
    )

    y_history = np.array([np.nan, np.inf])
    y_new = np.array([np.nan, np.inf])

    y_clean = opt._apply_penalty_NA(y_new, y_history=y_history)

    # Should still work with default large penalty
    assert np.all(np.isfinite(y_clean))
    assert np.all(y_clean > 1e9)  # Large default penalty


def test_apply_penalty_no_nan_values():
    """Test when y has no NaN/inf values."""
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1), bounds=[(-5, 5), (-5, 5)], verbose=False
    )

    y_history = np.array([1.0, 2.0, 3.0])
    y_new = np.array([4.0, 5.0, 6.0])

    y_clean = opt._apply_penalty_NA(y_new, y_history=y_history)

    # Should return unchanged
    assert np.array_equal(y_clean, y_new)


def test_apply_penalty_random_noise_uniqueness():
    """Test that random noise makes penalty values unique."""
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1),
        bounds=[(-5, 5), (-5, 5)],
        seed=42,  # For reproducibility
        verbose=False,
    )

    y_history = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_new = np.array([np.nan, np.nan, np.nan, np.nan])

    y_clean = opt._apply_penalty_NA(y_new, y_history=y_history)

    # All values should be finite but slightly different due to noise
    assert np.all(np.isfinite(y_clean))

    # Check that not all penalty values are identical
    unique_values = np.unique(y_clean)
    assert len(unique_values) > 1, "Penalty values should have random noise"


def test_apply_penalty_with_self_penalty_attribute():
    """Test that self.penalty is used as fallback when provided."""
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1),
        bounds=[(-5, 5), (-5, 5)],
        penalty=True,
        penalty_val=999.0,
        verbose=False,
    )

    # Only one finite value in history - should trigger fallback
    y_history = np.array([5.0])
    y_new = np.array([np.nan, np.inf])

    y_clean = opt._apply_penalty_NA(y_new, y_history=y_history)

    assert np.all(np.isfinite(y_clean))

    # Should use self.penalty = 999.0
    assert 998.0 < y_clean[0] < 1000.0
    assert 998.0 < y_clean[1] < 1000.0


def test_apply_penalty_custom_sd():
    """Test with custom standard deviation for noise."""
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1), bounds=[(-5, 5), (-5, 5)], verbose=False
    )

    y_history = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_new = np.array([np.nan] * 100)  # Many NaN values

    # Use larger sd for more spread
    y_clean = opt._apply_penalty_NA(y_new, y_history=y_history, sd=1.0)

    assert np.all(np.isfinite(y_clean))

    # Check that values have reasonable spread (sd=1.0)
    assert np.std(y_clean) > 0.5  # Should have noticeable variance


def test_apply_penalty_integration_with_optimize():
    """Integration test: verify penalty is computed from history during optimization."""
    call_count = [0]

    def objective_with_nan(X):
        """Function that returns NaN on certain calls."""
        results = []
        for x in X:
            call_count[0] += 1
            # Return NaN on specific calls (after initial design)
            if 15 <= call_count[0] <= 17:
                results.append(np.nan)
            else:
                results.append(np.sum(x**2))
        return np.array(results)

    optimizer = SpotOptim(
        fun=objective_with_nan,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=20,
        n_initial=10,
        seed=42,
        penalty=True,
        verbose=False,
    )

    result = optimizer.optimize()

    # Optimization should complete successfully
    assert result.success is True

    # All stored y values should be finite (penalties applied)
    assert np.all(np.isfinite(optimizer.y_))

    # Should have completed 20 evaluations
    assert len(optimizer.y_) == 20

    # Penalty values (from NaN) should be larger than typical objective values
    # Initial design should have values around 0-50 (sum of squares in [-5,5])
    # Penalties should be much larger
    max_initial = np.max(optimizer.y_[:10])
    nan_penalties = optimizer.y_[14:17]  # These were NaN
    assert np.all(nan_penalties > max_initial)


def test_apply_penalty_verbose_output(capsys):
    """Test verbose output messages."""
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1), bounds=[(-5, 5), (-5, 5)], verbose=True
    )

    y_history = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_new = np.array([np.nan, 6.0, np.inf])

    opt._apply_penalty_NA(y_new, y_history=y_history)

    captured = capsys.readouterr()
    assert "Warning: Found 2 NaN/inf value(s)" in captured.out
    assert "adaptive penalty" in captured.out


def test_apply_penalty_preserves_array_shape():
    """Test that output array has same shape as input."""
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1), bounds=[(-5, 5), (-5, 5)], verbose=False
    )

    y_history = np.array([1.0, 2.0, 3.0])

    # Test various shapes
    for shape in [(5,), (10,), (1,), (100,)]:
        y_new = np.full(shape, np.nan)
        y_clean = opt._apply_penalty_NA(y_new, y_history=y_history)
        assert y_clean.shape == shape
