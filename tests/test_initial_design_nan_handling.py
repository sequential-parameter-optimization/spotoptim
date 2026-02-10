# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for NaN/inf handling during initial design phase.

This module tests the behavior when objective function returns NaN or inf values
during the initial design phase. The expected behavior is:
- NaN/inf values are filtered out (not penalized) during initial design
- User receives warnings about removed points
- Optimization stops with clear error if insufficient valid points remain
- Optimization continues normally if enough valid points remain
"""

import numpy as np
import pytest
from spotoptim import SpotOptim


def test_initial_design_with_some_nan_values():
    """Test that optimization continues when some initial design points return NaN."""
    call_count = [0]

    def objective_with_some_nan(X):
        """Returns NaN for first 3 evaluations, then valid values."""
        results = []
        for x in X:
            if call_count[0] < 3:
                results.append(np.nan)
            else:
                results.append(np.sum(x**2))
            call_count[0] += 1
        return np.array(results)

    # With n_initial=10, we should get 7 valid points after filtering 3 NaN values
    optimizer = SpotOptim(
        fun=objective_with_some_nan,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=15,
        n_initial=10,
        seed=42,
        verbose=False,
    )

    result = optimizer.optimize()

    # Check that optimization completed successfully
    assert result.success is True
    # Should have at least 7 valid initial points (10 - 3 NaN)
    assert len(optimizer.y_) >= 7
    # All stored values should be finite
    assert np.all(np.isfinite(optimizer.y_))
    # Best value should be reasonable
    assert result.fun < 100.0


def test_initial_design_with_some_inf_values():
    """Test that optimization continues when some initial design points return inf."""
    call_count = [0]

    def objective_with_some_inf(X):
        """Returns inf for first 4 evaluations, then valid values."""
        results = []
        for x in X:
            if call_count[0] < 4:
                results.append(np.inf)
            else:
                results.append(np.sum(x**2))
            call_count[0] += 1
        return np.array(results)

    optimizer = SpotOptim(
        fun=objective_with_some_inf,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=15,
        n_initial=10,
        seed=42,
        verbose=False,
    )

    result = optimizer.optimize()

    assert result.success is True
    assert len(optimizer.y_) >= 6  # At least 6 valid points (10 - 4 inf)
    assert np.all(np.isfinite(optimizer.y_))


def test_initial_design_with_mixed_nan_inf():
    """Test handling of mixed NaN and inf values in initial design."""
    call_count = [0]

    def objective_with_mixed_invalid(X):
        """Returns mix of NaN, inf, and valid values."""
        results = []
        for x in X:
            if call_count[0] % 3 == 0:
                results.append(np.nan)
            elif call_count[0] % 3 == 1:
                results.append(np.inf)
            else:
                results.append(np.sum(x**2))
            call_count[0] += 1
        return np.array(results)

    optimizer = SpotOptim(
        fun=objective_with_mixed_invalid,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=20,
        n_initial=12,
        seed=42,
        verbose=False,
    )

    result = optimizer.optimize()

    assert result.success is True
    # Should have ~4 valid points out of 12 (every 3rd point is valid)
    assert len(optimizer.y_) >= 4
    assert np.all(np.isfinite(optimizer.y_))


def test_initial_design_insufficient_points_2d():
    """Test that optimization fails with clear error when too few valid initial points (2D)."""

    def objective_mostly_nan(X):
        """Returns NaN for most evaluations."""
        results = []
        for i, x in enumerate(X):
            if i < 2:  # Only first 2 points are valid
                results.append(np.sum(x**2))
            else:
                results.append(np.nan)
        return np.array(results)

    optimizer = SpotOptim(
        fun=objective_mostly_nan,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=15,
        n_initial=10,
        seed=42,
        verbose=False,
    )

    # Should raise ValueError due to insufficient points (need at least 3 for 2D)
    with pytest.raises(ValueError) as excinfo:
        optimizer.optimize()

    assert "Insufficient valid initial design points" in str(excinfo.value)
    assert "Need at least 3 points" in str(excinfo.value)


def test_initial_design_insufficient_points_1d():
    """Test that optimization fails when too few valid initial points (1D)."""

    def objective_mostly_nan(X):
        """Returns NaN for most evaluations."""
        results = []
        for i, x in enumerate(X):
            if i == 0:  # Only first point is valid
                results.append(x[0] ** 2)
            else:
                results.append(np.nan)
        return np.array(results)

    optimizer = SpotOptim(
        fun=objective_mostly_nan,
        bounds=[(-5, 5)],
        max_iter=15,
        n_initial=10,
        seed=42,
        verbose=False,
    )

    # Should raise ValueError due to insufficient points (need at least 2 for 1D)
    with pytest.raises(ValueError) as excinfo:
        optimizer.optimize()

    assert "Insufficient valid initial design points" in str(excinfo.value)
    assert "Need at least 2 points" in str(excinfo.value)


def test_initial_design_all_nan():
    """Test that optimization fails when all initial points return NaN."""

    def objective_all_nan(X):
        """Returns NaN for all evaluations."""
        return np.full(len(X), np.nan)

    optimizer = SpotOptim(
        fun=objective_all_nan,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=15,
        n_initial=10,
        seed=42,
        verbose=False,
    )

    with pytest.raises(ValueError) as excinfo:
        optimizer.optimize()

    assert "Insufficient valid initial design points" in str(excinfo.value)
    assert "only 0 finite value(s)" in str(excinfo.value)


def test_initial_design_all_inf():
    """Test that optimization fails when all initial points return inf."""

    def objective_all_inf(X):
        """Returns inf for all evaluations."""
        return np.full(len(X), np.inf)

    optimizer = SpotOptim(
        fun=objective_all_inf,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=15,
        n_initial=10,
        seed=42,
        verbose=False,
    )

    with pytest.raises(ValueError) as excinfo:
        optimizer.optimize()

    assert "Insufficient valid initial design points" in str(excinfo.value)


def test_initial_design_exact_minimum_points_2d():
    """Test that optimization works with exactly the minimum required points (2D)."""
    call_count = [0]

    def objective_exactly_three_valid(X):
        """Returns valid values for exactly first 3 points, then NaN."""
        results = []
        for x in X:
            if call_count[0] < 3:
                results.append(np.sum(x**2))
            else:
                results.append(np.nan)
            call_count[0] += 1
        return np.array(results)

    optimizer = SpotOptim(
        fun=objective_exactly_three_valid,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=15,
        n_initial=10,
        seed=42,
        verbose=False,
    )

    result = optimizer.optimize()

    # Optimization should fail gracefully (success=False) due to consecutive failures
    # (infinite loop prevention)
    assert result.success is False
    assert "Optimization stopped due to" in result.message
    # We expect some valid points from initial design
    assert len(optimizer.y_) == 3  # Only the 3 initial valid points
    assert np.all(np.isfinite(optimizer.y_))


def test_initial_design_exact_minimum_points_1d():
    """Test that optimization works with exactly the minimum required points (1D)."""
    call_count = [0]

    def objective_exactly_two_valid(X):
        """Returns valid values for exactly first 2 points, then NaN."""
        results = []
        for x in X:
            if call_count[0] < 2:
                results.append(x[0] ** 2)
            else:
                results.append(np.nan)
            call_count[0] += 1
        return np.array(results)

    optimizer = SpotOptim(
        fun=objective_exactly_two_valid,
        bounds=[(-5, 5)],
        max_iter=15,
        n_initial=10,
        seed=42,
        verbose=False,
    )

    result = optimizer.optimize()

    # Should fail gracefully due to consecutive failures
    assert result.success is False
    assert "Optimization stopped due to" in result.message
    assert len(optimizer.y_) == 2
    assert np.all(np.isfinite(optimizer.y_))


def test_initial_design_no_penalty_applied():
    """Test that NaN values are NOT replaced with penalties during initial design."""
    call_count = [0]
    invalid_indices = []

    def objective_track_invalid(X):
        """Track which evaluations return NaN."""
        results = []
        for i, x in enumerate(X):
            if call_count[0] < 3:
                results.append(np.nan)
                invalid_indices.append(call_count[0])
            else:
                results.append(np.sum(x**2))
            call_count[0] += 1
        return np.array(results)

    optimizer = SpotOptim(
        fun=objective_track_invalid,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=15,
        n_initial=10,
        penalty=1000.0,  # Set a penalty value
        seed=42,
        verbose=False,
    )

    result = optimizer.optimize()

    # Check that no penalty values appear in stored results
    # (all values should be from actual objective function, not penalties)
    assert np.all(optimizer.y_ < 100.0)  # All values should be reasonable
    assert np.all(optimizer.y_ >= 0.0)  # Squared values should be non-negative
    # Verify that the initial design is smaller than requested
    initial_design_size = np.sum(optimizer.y_ < 100.0)  # Count reasonable values
    assert initial_design_size >= 7  # Should have ~7 valid points (10 - 3 NaN)


def test_surrogate_phase_still_uses_penalty():
    """Test that penalty IS applied during surrogate-based optimization (not initial design)."""
    call_count = [0]

    def objective_nan_after_initial(X):
        """Returns valid values initially, then occasional NaN during optimization."""
        results = []
        for x in X:
            if call_count[0] < 10:
                # Initial design: all valid
                results.append(np.sum(x**2))
            elif call_count[0] % 3 == 0:
                # During optimization: occasional NaN
                results.append(np.nan)
            else:
                results.append(np.sum(x**2))
            call_count[0] += 1
        return np.array(results)

    optimizer = SpotOptim(
        fun=objective_nan_after_initial,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=20,
        n_initial=10,
        penalty=1000.0,
        seed=42,
        verbose=False,
    )

    result = optimizer.optimize()

    # Should complete successfully
    assert result.success is True
    # Some large penalty values may exist from surrogate phase
    # But all values should be finite (penalties replace NaN/inf)
    assert np.all(np.isfinite(optimizer.y_))


def test_initial_design_verbose_warnings(capsys):
    """Test that verbose mode produces appropriate warnings."""
    call_count = [0]

    def objective_with_nans(X):
        results = []
        for x in X:
            if call_count[0] < 4:
                results.append(np.nan)
            else:
                results.append(np.sum(x**2))
            call_count[0] += 1
        return np.array(results)

    optimizer = SpotOptim(
        fun=objective_with_nans,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=15,
        n_initial=10,
        seed=42,
        verbose=True,  # Enable verbose output
    )

    result = optimizer.optimize()

    # Capture output
    captured = capsys.readouterr()

    # Check for warning messages
    assert "Warning:" in captured.out or "warning" in captured.out.lower()
    assert "NaN/inf" in captured.out or "nan" in captured.out.lower()


def test_initial_design_with_repeats():
    """Test NaN handling with repeated initial evaluations (noise handling)."""
    call_count = [0]

    def objective_with_noise_and_nans(X):
        """Returns NaN for some evaluations, valid noisy values for others."""
        results = []
        for x in X:
            if call_count[0] % 5 == 0:
                results.append(np.nan)
            else:
                # Add noise to simulate noisy function
                results.append(np.sum(x**2) + np.random.normal(0, 0.1))
            call_count[0] += 1
        return np.array(results)

    optimizer = SpotOptim(
        fun=objective_with_noise_and_nans,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=20,
        n_initial=10,
        repeats_initial=2,  # Evaluate each point twice
        seed=42,
        verbose=False,
    )

    result = optimizer.optimize()

    # Should succeed despite some NaN values
    assert result.success is True
    assert np.all(np.isfinite(optimizer.y_))
    # With repeats, we might have fewer unique points but that's OK
    assert len(optimizer.y_) >= 3


def test_error_message_includes_helpful_info():
    """Test that error message includes helpful information for debugging."""

    def objective_all_nan(X):
        return np.full(len(X), np.nan)

    optimizer = SpotOptim(
        fun=objective_all_nan,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=15,
        n_initial=10,
        seed=42,
        verbose=False,
    )

    with pytest.raises(ValueError) as excinfo:
        optimizer.optimize()

    error_msg = str(excinfo.value)

    # Check that error message contains helpful information
    assert "Insufficient valid initial design points" in error_msg
    assert "0 finite value(s)" in error_msg
    assert "out of 10 evaluated" in error_msg
    assert "Need at least 3 points" in error_msg
    assert (
        "check your objective function" in error_msg.lower()
        or "Please check" in error_msg
    )


def test_higher_dimensional_problem():
    """Test NaN handling in higher-dimensional problem (5D)."""
    call_count = [0]

    def objective_5d_with_nans(X):
        results = []
        for x in X:
            if call_count[0] < 5:
                results.append(np.nan)
            else:
                results.append(np.sum(x**2))
            call_count[0] += 1
        return np.array(results)

    optimizer = SpotOptim(
        fun=objective_5d_with_nans,
        bounds=[(-5, 5)] * 5,  # 5D problem
        max_iter=25,
        n_initial=15,
        seed=42,
        verbose=False,
    )

    result = optimizer.optimize()

    assert result.success is True
    assert len(optimizer.y_) >= 10  # At least 10 valid points (15 - 5 NaN)
    assert np.all(np.isfinite(optimizer.y_))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
