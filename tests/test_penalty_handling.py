# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for penalty handling functionality (_apply_penalty_NA method).

This test suite validates the _apply_penalty_NA() method, ensuring:
- Adaptive penalty calculation (max + 3*std) when penalty_value is None
- Correct handling of NaN and inf values
- Fallback behavior with insufficient finite values
- Custom penalty values work correctly
- Random noise is added to avoid identical penalty values
"""

import numpy as np
from spotoptim.SpotOptim import SpotOptim


class TestApplyPenaltyNABasic:
    """Test suite for basic _apply_penalty_NA functionality."""

    def test_no_nan_values_returns_unchanged_array(self):
        """Test that array without NaN/inf values is returned unchanged."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            seed=42,
        )

        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_result = optimizer._apply_penalty_NA(y)

        np.testing.assert_array_equal(y, y_result)

    def test_all_finite_values_remain_unchanged(self):
        """Test that finite values (including zeros and negatives) remain unchanged."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5,
            seed=42,
        )

        y = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        y_result = optimizer._apply_penalty_NA(y)

        np.testing.assert_array_equal(y, y_result)


class TestApplyPenaltyNAAdaptive:
    """Test suite for adaptive penalty calculation (max + 3*std)."""

    def test_adaptive_penalty_with_single_nan(self):
        """Test adaptive penalty with one NaN value."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5,
            seed=42,
            verbose=False,
        )

        y = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
        y_result = optimizer._apply_penalty_NA(y)

        # Check all values are finite
        assert np.all(np.isfinite(y_result))

        # First 4 values should be unchanged
        np.testing.assert_array_equal(y_result[:4], y[:4])

        # NaN should be replaced with value > max (adaptive penalty)
        # Expected: max(1,2,3,4) + 3*std(1,2,3,4) = 4 + 3*1.291 ≈ 7.87
        max_y = np.max(y[:4])
        std_y = np.std(y[:4], ddof=1)
        expected_penalty = max_y + 3.0 * std_y

        # The penalty value should be close to expected (with small random noise)
        assert y_result[4] > max_y  # Should be larger than max
        assert (
            np.abs(y_result[4] - expected_penalty) < 1.0
        )  # Within reasonable range of noise

    def test_adaptive_penalty_with_multiple_nans(self):
        """Test adaptive penalty with multiple NaN values."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5,
            seed=42,
            verbose=False,
        )

        y = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0])
        y_result = optimizer._apply_penalty_NA(y)

        # Check all values are finite
        assert np.all(np.isfinite(y_result))

        # Finite values should be unchanged
        assert y_result[0] == 1.0
        assert y_result[1] == 2.0
        assert y_result[3] == 4.0
        assert y_result[5] == 6.0

        # NaN values should be replaced with penalty values
        max_y = np.max([1.0, 2.0, 4.0, 6.0])
        assert y_result[2] > max_y
        assert y_result[4] > max_y

        # The two penalty values should be different (due to random noise)
        assert y_result[2] != y_result[4]

    def test_adaptive_penalty_with_inf_values(self):
        """Test adaptive penalty with inf values."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5,
            seed=42,
            verbose=False,
        )

        y = np.array([1.0, 2.0, 3.0, np.inf, -np.inf])
        y_result = optimizer._apply_penalty_NA(y)

        # Check all values are finite
        assert np.all(np.isfinite(y_result))

        # First 3 values should be unchanged
        np.testing.assert_array_equal(y_result[:3], y[:3])

        # Inf values should be replaced
        max_y = np.max(y[:3])
        assert y_result[3] > max_y
        assert y_result[4] > max_y

    def test_adaptive_penalty_calculation_correctness(self):
        """Test that adaptive penalty is calculated correctly as max + 3*std."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        # Use fixed seed for reproducibility
        np.random.seed(42)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5,
            seed=42,
            verbose=False,
        )

        y = np.array([10.0, 20.0, 30.0, 40.0, np.nan])
        y_result = optimizer._apply_penalty_NA(y, sd=0.0)  # No noise for exact test

        # Calculate expected penalty
        finite_values = y[:4]
        max_y = np.max(finite_values)  # 40.0
        std_y = np.std(finite_values, ddof=1)  # ~12.91
        expected_penalty = max_y + 3.0 * std_y  # 40.0 + 38.73 ≈ 78.73

        # Check the penalty value (should be exact with sd=0.0)
        assert np.abs(y_result[4] - expected_penalty) < 0.01


class TestApplyPenaltyNAFallback:
    """Test suite for fallback behavior with insufficient finite values."""

    def test_fallback_with_all_nan(self):
        """Test fallback to self.penalty when all values are NaN."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5,
            penalty=True,
            penalty_val=1000.0,
            seed=42,
            verbose=False,
        )

        y = np.array([np.nan, np.nan, np.nan])
        y_result = optimizer._apply_penalty_NA(y, sd=0.0)

        # All values should be replaced with self.penalty (no noise for test)
        np.testing.assert_allclose(y_result, [1000.0, 1000.0, 1000.0], atol=0.01)

    def test_fallback_with_single_finite_value(self):
        """Test fallback to self.penalty when only one finite value exists."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5,
            penalty=True,
            penalty_val=500.0,
            seed=42,
            verbose=False,
        )

        y = np.array([5.0, np.nan, np.nan, np.nan])
        y_result = optimizer._apply_penalty_NA(y, sd=0.0)

        # First value unchanged
        assert y_result[0] == 5.0

        # NaN values should use self.penalty (fallback)
        assert np.allclose(y_result[1:], [500.0, 500.0, 500.0], atol=0.01)

    def test_fallback_when_penalty_is_inf(self):
        """Test fallback behavior when self.penalty is np.inf."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5,
            penalty=True,
            penalty_val=np.inf,  # Default penalty
            seed=42,
            verbose=False,
        )

        y = np.array([np.nan, np.nan])
        y_result = optimizer._apply_penalty_NA(y, sd=0.0)

        # Should fallback to np.inf when no finite values available
        assert np.all(np.isinf(y_result))


class TestApplyPenaltyNACustomPenalty:
    """Test suite for custom penalty_value parameter."""

    def test_custom_penalty_value(self):
        """Test that custom penalty_value is used when provided."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5,
            seed=42,
            verbose=False,
        )

        y = np.array([1.0, 2.0, np.nan, 4.0])
        custom_penalty = 999.0
        y_result = optimizer._apply_penalty_NA(y, penalty_value=custom_penalty, sd=0.0)

        # Check finite values unchanged
        assert y_result[0] == 1.0
        assert y_result[1] == 2.0
        assert y_result[3] == 4.0

        # NaN should be replaced with custom penalty (no noise)
        assert np.abs(y_result[2] - custom_penalty) < 0.01

    def test_custom_penalty_overrides_adaptive(self):
        """Test that custom penalty_value overrides adaptive calculation."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5,
            seed=42,
            verbose=False,
        )

        y = np.array([10.0, 20.0, 30.0, np.nan])
        custom_penalty = 50.0  # Much smaller than adaptive would be
        y_result = optimizer._apply_penalty_NA(y, penalty_value=custom_penalty, sd=0.0)

        # Should use custom penalty, not adaptive
        assert np.abs(y_result[3] - custom_penalty) < 0.01

        # Verify it's NOT using adaptive penalty
        max_y = np.max(y[:3])
        std_y = np.std(y[:3], ddof=1)
        adaptive_penalty = max_y + 3.0 * std_y
        assert np.abs(y_result[3] - adaptive_penalty) > 1.0  # Should be different


class TestApplyPenaltyNARandomNoise:
    """Test suite for random noise addition."""

    def test_random_noise_creates_unique_penalties(self):
        """Test that random noise creates unique penalty values."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5,
            penalty=True,
            penalty_val=100.0,  # Set finite penalty for test
            seed=42,
            verbose=False,
        )

        # Use at least 2 finite values for adaptive penalty
        y = np.array([1.0, 2.0, np.nan, np.nan, np.nan, np.nan])
        y_result = optimizer._apply_penalty_NA(y, sd=0.1)  # Default noise

        # All NaN values should be replaced
        assert np.all(np.isfinite(y_result))

        # First two values should be unchanged
        assert y_result[0] == 1.0
        assert y_result[1] == 2.0

        # The penalty values should be different (due to random noise)
        penalty_values = y_result[2:]
        assert len(set(penalty_values)) == len(penalty_values)  # All unique

    def test_zero_noise_creates_identical_penalties(self):
        """Test that sd=0.0 creates identical penalty values."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5,
            seed=42,
            verbose=False,
        )

        y = np.array([1.0, np.nan, np.nan, np.nan])
        y_result = optimizer._apply_penalty_NA(y, sd=0.0)  # No noise

        # All NaN values should have same penalty (no noise)
        penalty_values = y_result[1:]
        assert np.all(penalty_values == penalty_values[0])

    def test_custom_noise_sd(self):
        """Test that custom standard deviation for noise works."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        np.random.seed(42)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5,
            seed=42,
            verbose=False,
        )

        y = np.array([1.0, 2.0, 3.0, np.nan])

        # Test with larger noise
        y_result_large = optimizer._apply_penalty_NA(y, sd=1.0)

        # Reset random state
        np.random.seed(42)

        # Test with smaller noise
        y_result_small = optimizer._apply_penalty_NA(y, sd=0.01)

        # Both should have replaced the NaN
        assert np.isfinite(y_result_large[3])
        assert np.isfinite(y_result_small[3])

        # The noise magnitude should differ
        # (This is probabilistic, but with seed=42 it should be reliable)


class TestApplyPenaltyNAIntegration:
    """Test suite for integration scenarios."""

    def test_mixed_nan_and_inf_values(self):
        """Test handling of mixed NaN and inf values."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5,
            seed=42,
            verbose=False,
        )

        y = np.array([1.0, np.nan, 3.0, np.inf, 5.0, -np.inf])
        y_result = optimizer._apply_penalty_NA(y)

        # All values should be finite
        assert np.all(np.isfinite(y_result))

        # Finite values should be unchanged
        assert y_result[0] == 1.0
        assert y_result[2] == 3.0
        assert y_result[4] == 5.0

        # Non-finite values should be replaced with penalties
        max_y = 5.0
        assert y_result[1] > max_y
        assert y_result[3] > max_y
        assert y_result[5] > max_y

    def test_with_negative_values(self):
        """Test adaptive penalty with negative values."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5,
            seed=42,
            verbose=False,
        )

        y = np.array([-10.0, -5.0, 0.0, 5.0, np.nan])
        y_result = optimizer._apply_penalty_NA(y)

        # All values should be finite
        assert np.all(np.isfinite(y_result))

        # Negative values should be unchanged
        assert y_result[0] == -10.0
        assert y_result[1] == -5.0
        assert y_result[2] == 0.0
        assert y_result[3] == 5.0

        # NaN should be replaced with adaptive penalty
        # max = 5.0, std ≈ 6.45, penalty ≈ 5.0 + 19.36 ≈ 24.36
        assert y_result[4] > 5.0

    def test_preserves_original_array(self):
        """Test that original array is not modified."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5,
            seed=42,
            verbose=False,
        )

        y_original = np.array([1.0, 2.0, np.nan, 4.0])
        y_copy = y_original.copy()

        y_result = optimizer._apply_penalty_NA(y_original)

        # Original array should be unchanged (NaN still present)
        assert np.isnan(y_original[2])
        assert np.isnan(y_copy[2])

        # Result should have NaN replaced
        assert np.isfinite(y_result[2])


class TestApplyPenaltyNAVerboseOutput:
    """Test suite for verbose output."""

    def test_verbose_output_with_adaptive_penalty(self, capsys):
        """Test that verbose mode prints correct message for adaptive penalty."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5,
            seed=42,
            verbose=True,
        )

        y = np.array([1.0, 2.0, 3.0, np.nan])
        _ = optimizer._apply_penalty_NA(y)

        captured = capsys.readouterr()
        assert "adaptive penalty" in captured.out.lower()
        assert "1 NaN/inf value(s)" in captured.out

    def test_verbose_output_with_fallback(self, capsys):
        """Test that verbose mode prints correct message for fallback."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5,
            penalty=True,
            penalty_val=500.0,
            seed=42,
            verbose=True,
        )

        y = np.array([5.0, np.nan])
        _ = optimizer._apply_penalty_NA(y)

        captured = capsys.readouterr()
        assert "insufficient finite values" in captured.out.lower()
        assert "penalty_value = 500.0" in captured.out.lower()

    def test_verbose_output_with_custom_penalty(self, capsys):
        """Test that verbose mode prints correct message for custom penalty."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5,
            seed=42,
            verbose=True,
        )

        y = np.array([1.0, 2.0, np.nan])
        custom_penalty = 999.0
        _ = optimizer._apply_penalty_NA(y, penalty_value=custom_penalty)

        captured = capsys.readouterr()
        assert "999" in captured.out or "999.0" in captured.out


class TestApplyPenaltyNAEdgeCases:
    """Test suite for edge cases."""

    def test_empty_array(self):
        """Test handling of empty array."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5,
            seed=42,
            verbose=False,
        )

        y = np.array([])
        y_result = optimizer._apply_penalty_NA(y)

        assert len(y_result) == 0

    def test_single_nan_value(self):
        """Test handling of single NaN value array."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5,
            penalty=True,
            penalty_val=100.0,
            seed=42,
            verbose=False,
        )

        y = np.array([np.nan])
        y_result = optimizer._apply_penalty_NA(y, sd=0.0)

        # Should fallback to self.penalty
        assert np.isfinite(y_result[0])
        assert np.abs(y_result[0] - 100.0) < 0.01

    def test_large_array_performance(self):
        """Test that method handles large arrays efficiently."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5,
            seed=42,
            verbose=False,
        )

        # Create large array with some NaN values
        y = np.random.rand(10000)
        y[::100] = np.nan  # Every 100th value is NaN

        import time

        start = time.time()
        y_result = optimizer._apply_penalty_NA(y)
        elapsed = time.time() - start

        # Should complete quickly (< 1 second)
        assert elapsed < 1.0

        # All values should be finite
        assert np.all(np.isfinite(y_result))

        # Non-NaN values should be unchanged
        assert np.allclose(y_result[1], y[1])
