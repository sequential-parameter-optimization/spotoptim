# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for the optimize_acquisition_func method in SpotOptim.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from spotoptim.SpotOptim import SpotOptim


class TestOptimizeAcquisitionFunc:
    """Test suite for optimize_acquisition_func method."""

    def test_optimize_acquisition_func_returns_valid_point(self):
        """Test that optimize_acquisition_func returns a point within bounds."""
        
        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            seed=42,
            acquisition_fun_return_size=1,
        )

        # Mock _acquisition_function to avoid needing a trained surrogate
        # Just return sum of squares (simple convex function)
        optimizer._acquisition_function = MagicMock(side_effect=lambda x: np.sum(x**2))

        # Call method
        x_opt = optimizer.optimize_acquisition_func()

        # Check shape
        assert x_opt.shape == (2,)

        # Check bounds
        assert -5 <= x_opt[0] <= 5
        assert -5 <= x_opt[1] <= 5

    @patch("spotoptim.SpotOptim.differential_evolution")
    def test_optimize_acquisition_func_calls_differential_evolution(self, mock_de):
        """Test that optimize_acquisition_func calls differential_evolution with correct args."""
        
        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            seed=42,
            acquisition_fun_return_size=1,
        )

        # Mock return value of differential_evolution
        mock_result = MagicMock()
        mock_result.x = np.array([1.2, 3.4])
        mock_de.return_value = mock_result

        # Call method
        x_opt = optimizer.optimize_acquisition_func()

        # Check call arguments
        mock_de.assert_called_once()
        call_kwargs = mock_de.call_args[1]
        
        # Check specific arguments that are important
        assert call_kwargs["bounds"] == optimizer.bounds
        assert call_kwargs["seed"] == optimizer.rng
        assert call_kwargs["maxiter"] == 10000
        assert call_kwargs["func"] == optimizer._acquisition_function
        
        # Check return value
        np.testing.assert_array_equal(x_opt, np.array([1.2, 3.4]))

    def test_optimize_acquisition_func_respects_bounds(self):
        """Verify optimized point respects different bounds."""
        
        def simple_func(X):
            return np.sum(X**2, axis=1)

        # Use asymmetric bounds
        bounds = [(-10, -5), (5, 10)]
        optimizer = SpotOptim(
            fun=simple_func,
            bounds=bounds,
            max_iter=10,
            n_initial=5,
            seed=42,
            acquisition_fun_return_size=1,
        )

        # Mock acquisition function
        optimizer._acquisition_function = MagicMock(return_value=0.0)

        # Call method
        x_opt = optimizer.optimize_acquisition_func()

        # Check bounds
        assert -10 <= x_opt[0] <= -5
        assert 5 <= x_opt[1] <= 10
