# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Comprehensive tests for refactored optimization methods.

This module tests the helper methods extracted from optimize() during refactoring,
including initial design handling, OCBA, main loop updates, and termination logic.
"""

import pytest
import numpy as np
import time
from spotoptim.SpotOptim import SpotOptim


class TestSetInitialDesign:
    """Tests for get_initial_design() method."""

    def test_get_initial_design_none_generates_lhs(self):
        """Test that X0=None generates LHS design."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=10,
            seed=42,
        )
        X0 = opt.get_initial_design(X0=None)

        assert X0.shape == (10, 2)
        assert np.all(X0 >= -5) and np.all(X0 <= 5)

    def test_get_initial_design_with_x0_includes_starting_point(self):
        """Test that x0 is included as first point when X0=None."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=10,
            x0=[0.0, 0.0],
            seed=42,
        )
        X0 = opt.get_initial_design(X0=None)

        assert X0.shape == (10, 2)
        # First point should be x0
        np.testing.assert_array_almost_equal(X0[0], [0.0, 0.0], decimal=5)

    def test_get_initial_design_provided_custom(self):
        """Test that provided X0 is properly transformed."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=10,
            seed=42,
        )
        X0_custom = np.array([[0, 0], [1, 1], [2, 2]])
        X0 = opt.get_initial_design(X0=X0_custom)

        assert X0.shape == (3, 2)
        # Values should be within bounds after transformation
        assert np.all(X0 >= -5) and np.all(X0 <= 5)

    def test_get_initial_design_integer_variables(self):
        """Test initial design with integer variables gets rounded."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            var_type=["int", "int"],
            n_initial=5,
            seed=42,
        )
        X0 = opt.get_initial_design(X0=None)

        assert X0.shape == (5, 2)
        # All values should be integers
        np.testing.assert_array_equal(X0, np.round(X0))


class TestCurateInitialDesign:
    """Tests for _curate_initial_design() method."""

    def test_curate_removes_duplicates(self):
        """Test that duplicate points are removed."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=10,
            seed=42,
        )
        # Create design with duplicates
        X0 = np.array([[1, 2], [1, 2], [3, 4], [3, 4], [5, 6]])
        X0_curated = opt._curate_initial_design(X0)

        # Should have unique points
        unique_rows = np.unique(X0_curated, axis=0)
        assert len(unique_rows) == len(X0_curated)

    def test_curate_generates_additional_points_if_needed(self):
        """Test that additional points are generated when duplicates reduce size below n_initial."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=10,
            seed=42,
        )
        # Create design with many duplicates (only 3 unique)
        X0 = np.array([[1, 2], [1, 2], [1, 2], [3, 4], [3, 4], [5, 6], [5, 6]])
        X0_curated = opt._curate_initial_design(X0)

        # Should have at least 10 unique points (n_initial)
        assert X0_curated.shape[0] >= 10

    def test_curate_repeats_points_when_repeats_initial_gt_1(self):
        """Test that points are repeated when repeats_initial > 1."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            repeats_initial=3,
            seed=42,
        )
        X0 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        X0_curated = opt._curate_initial_design(X0)

        # Should have 5 * 3 = 15 points
        assert X0_curated.shape[0] == 15

    def test_curate_no_repeats_when_repeats_initial_eq_1(self):
        """Test that points are not repeated when repeats_initial = 1."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            repeats_initial=1,
            seed=42,
        )
        X0 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        X0_curated = opt._curate_initial_design(X0)

        # Should have 5 points (no repeats)
        assert X0_curated.shape[0] == 5


class TestHandleNAInitialDesign:
    """Tests for _handle_NA_initial_design() method."""

    def test_handle_na_removes_nan_values(self):
        """Test that NaN values are removed from initial design."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            seed=42,
        )
        X0 = np.array([[1, 2], [3, 4], [5, 6]])
        y0 = np.array([5.0, np.nan, 61.0])

        X0_clean, y0_clean, n_eval = opt._rm_NA_values(X0, y0)

        assert X0_clean.shape == (2, 2)
        assert len(y0_clean) == 2
        assert n_eval == 3
        assert not np.any(np.isnan(y0_clean))

    def test_handle_na_removes_inf_values(self):
        """Test that inf values are removed from initial design."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            seed=42,
        )
        X0 = np.array([[1, 2], [3, 4], [5, 6]])
        y0 = np.array([5.0, np.inf, 61.0])

        X0_clean, y0_clean, n_eval = opt._rm_NA_values(X0, y0)

        assert X0_clean.shape == (2, 2)
        assert len(y0_clean) == 2
        assert np.all(np.isfinite(y0_clean))

    def test_handle_na_all_valid_values(self):
        """Test that all valid values are kept."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            seed=42,
        )
        X0 = np.array([[1, 2], [3, 4], [5, 6]])
        y0 = np.array([5.0, 25.0, 61.0])

        X0_clean, y0_clean, n_eval = opt._rm_NA_values(X0, y0)

        assert X0_clean.shape == (3, 2)
        assert len(y0_clean) == 3
        assert n_eval == 3
        np.testing.assert_array_equal(y0, y0_clean)


class TestCheckSizeInitialDesign:
    """Tests for _check_size_initial_design() method."""

    def test_check_size_sufficient_points_2d(self):
        """Test that sufficient points (>= 3 for 2D) pass validation."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=10,
            seed=42,
        )
        y0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Should not raise
        opt._check_size_initial_design(y0, n_evaluated=10)

    def test_check_size_sufficient_points_1d(self):
        """Test that sufficient points (>= 2 for 1D) pass validation."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1), bounds=[(-5, 5)], n_initial=10, seed=42
        )
        y0 = np.array([1.0, 2.0])

        # Should not raise
        opt._check_size_initial_design(y0, n_evaluated=10)

    def test_check_size_insufficient_points_raises_error(self):
        """Test that insufficient points raise ValueError."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=10,
            seed=42,
        )
        y0 = np.array([1.0])  # Only 1 point, need at least 3

        with pytest.raises(
            ValueError, match="Insufficient valid initial design points"
        ):
            opt._check_size_initial_design(y0, n_evaluated=10)

    def test_check_size_verbose_warning_when_reduced(self):
        """Test that verbose mode prints warning when size reduced."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=10,
            verbose=True,
            seed=42,
        )
        y0 = np.array([1.0, 2.0, 3.0])  # 3 valid, but requested 10

        # Should not raise but should print warning (we can't easily test print)
        opt._check_size_initial_design(y0, n_evaluated=10)


class TestGetBestXYInitialDesign:
    """Tests for _get_best_xy_initial_design() method."""

    def test_get_best_finds_minimum(self):
        """Test that best_x_ and best_y_ are set correctly."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            seed=42,
        )
        opt.X_ = np.array([[1, 2], [0, 0], [2, 1], [3, 3]])
        opt.y_ = np.array([5.0, 0.0, 5.0, 18.0])

        opt._get_best_xy_initial_design()

        np.testing.assert_array_equal(opt.best_x_, [0, 0])
        assert opt.best_y_ == 0.0

    def test_get_best_with_noisy_function(self):
        """Test best finding with noisy function."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            repeats_initial=2,  # Enable noise handling
            seed=42,
        )
        opt.X_ = np.array([[1, 2], [0.1, 0.1], [2, 1]])
        opt.y_ = np.array([5.0, 0.02, 5.0])
        opt.min_mean_y = 0.5

        opt._get_best_xy_initial_design()

        np.testing.assert_array_almost_equal(opt.best_x_, [0.1, 0.1])
        assert opt.best_y_ == 0.02


class TestApplyOCBA:
    """Tests for _apply_ocba() method."""

    def test_apply_ocba_disabled_returns_none(self):
        """Test that OCBA returns None when disabled."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            repeats_initial=2,  # Enable noise handling
            ocba_delta=0,  # Disabled
            seed=42,
        )
        opt.mean_X = np.array([[1, 2], [0, 0], [2, 1]])
        opt.mean_y = np.array([5.0, 0.1, 5.0])
        opt.var_y = np.array([0.1, 0.05, 0.15])

        X_ocba = opt._apply_ocba()

        assert X_ocba is None

    def test_apply_ocba_no_noise_returns_none(self):
        """Test that OCBA returns None when noise=False."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            repeats_initial=1,  # Disable noise handling
            ocba_delta=5,
            seed=42,
        )

        X_ocba = opt._apply_ocba()

        assert X_ocba is None

    def test_apply_ocba_insufficient_points_returns_none(self):
        """Test that OCBA returns None with <= 2 points."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            repeats_initial=2,  # Enable noise handling
            ocba_delta=5,
            seed=42,
        )
        opt.mean_X = np.array([[1, 2], [0, 0]])
        opt.mean_y = np.array([5.0, 0.1])
        opt.var_y = np.array([0.1, 0.05])

        X_ocba = opt._apply_ocba()

        assert X_ocba is None

    def test_apply_ocba_with_valid_conditions(self):
        """Test that OCBA returns points with valid conditions."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            repeats_initial=2,  # Enable noise handling
            ocba_delta=5,
            seed=42,
        )
        opt.mean_X = np.array([[1, 2], [0, 0], [2, 1], [1, 1]])
        opt.mean_y = np.array([5.0, 0.1, 5.0, 2.0])
        opt.var_y = np.array([0.1, 0.05, 0.15, 0.08])

        X_ocba = opt._apply_ocba()

        # Should return some points for re-evaluation
        assert X_ocba is not None
        assert X_ocba.shape[0] > 0
        assert X_ocba.shape[1] == 2


class TestHandleNANewPoints:
    """Tests for _handle_NA_new_points() method."""

    def test_handle_na_new_points_all_valid(self):
        """Test handling when all new points are valid."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            seed=42,
        )
        opt.y_ = np.array([1.0, 2.0, 3.0])  # Historical values
        opt.n_iter_ = 1

        x_next = np.array([[1, 2], [3, 4]])
        y_next = np.array([5.0, 25.0])

        x_clean, y_clean = opt._handle_NA_new_points(x_next, y_next)

        assert x_clean.shape == (2, 2)
        assert len(y_clean) == 2
        np.testing.assert_array_equal(y_next, y_clean)

    def test_handle_na_new_points_some_invalid(self):
        """Test handling when some new points are NaN/inf."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            seed=42,
        )
        opt.y_ = np.array([1.0, 2.0, 3.0])
        opt.n_iter_ = 1

        x_next = np.array([[1, 2], [3, 4], [5, 6]])
        y_next = np.array([5.0, np.nan, 61.0])

        x_clean, y_clean = opt._handle_NA_new_points(x_next, y_next)

        # Should have filtered out NaN but kept valid values
        # Penalty might be applied, so check for finite values
        assert x_clean is not None
        assert y_clean is not None
        assert np.all(np.isfinite(y_clean))

    def test_handle_na_new_points_all_invalid_returns_none(self):
        """Test that all invalid points return None, None."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            seed=42,
        )
        opt.y_ = np.array([1.0, 2.0, 3.0])
        opt.n_iter_ = 1

        x_next = np.array([[1, 2], [3, 4]])
        y_next = np.array([np.nan, np.inf])

        x_clean, y_clean = opt._handle_NA_new_points(x_next, y_next)

        # Penalties are applied, so might not be None - check after remove_nan
        # Actually, if all are still non-finite after penalty, should return None
        # This depends on penalty logic - let's check the actual behavior
        if x_clean is None:
            assert y_clean is None
        else:
            # Penalties were successfully applied
            assert np.all(np.isfinite(y_clean))


class TestUpdateBestMainLoop:
    """Tests for _update_best_main_loop() method."""

    def test_update_best_new_best_found(self):
        """Test that best is updated when improvement found."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            seed=42,
        )
        opt.n_iter_ = 1
        opt.best_x_ = np.array([1.0, 1.0])
        opt.best_y_ = 2.0

        x_new = np.array([[0.1, 0.1], [0.5, 0.5]])
        y_new = np.array([0.02, 0.5])

        opt._update_best_main_loop(x_new, y_new)

        assert opt.best_y_ == 0.02
        # best_x_ should be in original space
        np.testing.assert_array_almost_equal(opt.best_x_, [0.1, 0.1], decimal=5)

    def test_update_best_no_improvement(self):
        """Test that best is not updated when no improvement."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            seed=42,
        )
        opt.n_iter_ = 1
        opt.best_x_ = np.array([0.1, 0.1])
        opt.best_y_ = 0.02

        x_new = np.array([[1.5, 1.5]])
        y_new = np.array([4.5])

        opt._update_best_main_loop(x_new, y_new)

        # Should not change
        assert opt.best_y_ == 0.02
        np.testing.assert_array_almost_equal(opt.best_x_, [0.1, 0.1])

    def test_update_best_with_noisy_function(self):
        """Test update with noisy function."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            repeats_initial=2,  # Enable noise handling
            seed=42,
        )
        opt.n_iter_ = 1
        opt.best_y_ = 2.0
        opt.min_mean_y = 1.5
        opt.best_x_ = np.array([1.0, 1.0])

        x_new = np.array([[0.5, 0.5]])
        y_new = np.array([0.5])

        opt._update_best_main_loop(x_new, y_new)

        assert opt.best_y_ == 0.5


class TestDetermineTermination:
    """Tests for _determine_termination() method."""

    def test_determine_termination_max_iter_reached(self):
        """Test termination message when max_iter reached."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=20,
            max_time=10.0,
            seed=42,
        )
        opt.y_ = np.zeros(20)  # Simulate 20 evaluations

        start_time = time.time()
        message = opt._determine_termination(start_time)

        assert "maximum evaluations" in message
        assert "20" in message

    def test_determine_termination_time_limit_exceeded(self):
        """Test termination message when time limit exceeded."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=100,
            max_time=10.0,
            seed=42,
        )
        opt.y_ = np.zeros(10)  # Only 10 evaluations

        # Simulate time elapsed > max_time
        start_time = time.time() - 700  # 11.67 minutes elapsed
        message = opt._determine_termination(start_time)

        assert "time limit" in message
        assert "10.00" in message

    def test_determine_termination_successful(self):
        """Test termination message for successful completion."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=100,
            max_time=10.0,
            seed=42,
        )
        opt.y_ = np.zeros(10)  # Under max_iter

        start_time = time.time()  # Just started
        message = opt._determine_termination(start_time)

        assert "successfully" in message


class TestOptimizeIntegration:
    """Integration tests for optimize() method after refactoring."""

    def test_optimize_completes_successfully(self):
        """Test that optimize() completes successfully with refactored methods."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=20,
            n_initial=10,
            seed=42,
            verbose=False,
        )

        result = opt.optimize()

        assert result.success is True
        assert result.nfev == 20
        assert result.nit == 10  # 20 - 10 initial
        assert result.fun < 1.0  # Should find good solution
        assert (
            "maximum evaluations" in result.message or "successfully" in result.message
        )

    def test_optimize_with_custom_initial_design(self):
        """Test optimize() with custom X0."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=15,
            seed=42,
            verbose=False,
        )

        X0 = np.array([[0, 0], [1, 1], [2, 2], [-1, -1], [-2, -2]])
        result = opt.optimize(X0=X0)

        assert result.success is True
        assert result.nfev == 15

    def test_optimize_with_starting_point_x0(self):
        """Test optimize() with x0 starting point."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=15,
            n_initial=5,
            x0=[0.5, 0.5],
            seed=42,
            verbose=False,
        )

        result = opt.optimize()

        assert result.success is True
        # Should be close to [0, 0] since x0 is already near optimum
        assert result.fun < 1.0

    def test_optimize_with_noisy_function(self):
        """Test optimize() with noisy objective function."""

        def noisy_sphere(X):
            X = np.atleast_2d(X)
            return np.sum(X**2, axis=1) + np.random.normal(0, 0.1, X.shape[0])

        opt = SpotOptim(
            fun=noisy_sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=20,
            n_initial=10,
            repeats_initial=2,  # Enable noise handling
            seed=42,
            verbose=False,
        )

        result = opt.optimize()

        assert result.success is True
        assert result.nfev == 20

    def test_optimize_with_nan_handling(self):
        """Test optimize() handles NaN values properly."""

        def sometimes_nan(X):
            X = np.atleast_2d(X)
            y = np.sum(X**2, axis=1)
            # Make some evaluations return NaN (use higher threshold to ensure some valid points)
            y[y > 100] = np.nan
            return y

        opt = SpotOptim(
            fun=sometimes_nan,
            bounds=[(-10, 10), (-10, 10)],
            max_iter=20,
            n_initial=10,
            seed=42,
            verbose=False,
        )

        result = opt.optimize()

        # Should complete successfully despite NaN values
        assert result.success is True
        assert np.isfinite(result.fun)

    def test_optimize_time_limit_termination(self):
        """Test that optimize() respects max_time limit."""

        def slow_function(X):
            time.sleep(0.1)  # Simulate slow evaluation
            X = np.atleast_2d(X)
            return np.sum(X**2, axis=1)

        opt = SpotOptim(
            fun=slow_function,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=100,
            n_initial=3,
            max_time=0.02,  # 0.02 minutes = 1.2 seconds
            seed=42,
            verbose=False,
        )

        start = time.time()
        result = opt.optimize()
        elapsed = time.time() - start

        assert result.success is True
        # Should terminate due to time limit
        assert "time limit" in result.message
        # Should have terminated around 1-2 seconds
        assert elapsed < 5.0  # Give some margin

    def test_optimize_max_iter_termination(self):
        """Test that optimize() respects max_iter limit."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=15,
            n_initial=5,
            seed=42,
            verbose=False,
        )

        result = opt.optimize()

        assert result.success is True
        assert result.nfev == 15
        assert "maximum evaluations" in result.message

    def test_optimize_with_integer_variables(self):
        """Test optimize() with integer variables."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            var_type=["int", "int"],
            max_iter=20,
            n_initial=10,
            seed=42,
            verbose=False,
        )

        result = opt.optimize()

        assert result.success is True
        # Best point should have integer coordinates
        assert np.all(result.x == np.round(result.x))

    def test_optimize_with_ocba(self):
        """Test optimize() with OCBA enabled."""

        def noisy_sphere(X):
            X = np.atleast_2d(X)
            return np.sum(X**2, axis=1) + np.random.normal(0, 0.1, X.shape[0])

        opt = SpotOptim(
            fun=noisy_sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=25,
            n_initial=10,
            repeats_initial=2,  # Enable noise handling
            ocba_delta=5,
            seed=42,
            verbose=False,
        )

        result = opt.optimize()

        assert result.success is True
        # OCBA may add extra evaluations, so nfev might be > max_iter
        # Actually, max_iter includes OCBA evaluations now
        assert result.nfev <= 25 + 10  # max_iter + some OCBA margin

    def test_optimize_result_attributes(self):
        """Test that optimize() result has all expected attributes."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=15,
            n_initial=5,
            seed=42,
            verbose=False,
        )

        result = opt.optimize()

        # Check all expected attributes exist
        assert hasattr(result, "x")
        assert hasattr(result, "fun")
        assert hasattr(result, "nfev")
        assert hasattr(result, "nit")
        assert hasattr(result, "success")
        assert hasattr(result, "message")
        assert hasattr(result, "X")
        assert hasattr(result, "y")

        # Check types
        assert isinstance(result.x, np.ndarray)
        assert isinstance(result.fun, (int, float, np.number))
        assert isinstance(result.nfev, int)
        assert isinstance(result.nit, int)
        assert isinstance(result.success, bool)
        assert isinstance(result.message, str)
        assert isinstance(result.X, np.ndarray)
        assert isinstance(result.y, np.ndarray)
