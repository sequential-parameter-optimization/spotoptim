# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Ackley function."""

import numpy as np
import pytest
from spotoptim.function import ackley


class TestAckleyBasic:
    """Test basic Ackley function functionality."""

    def test_ackley_global_minimum_2d(self):
        """Test that global minimum is at origin with value 0."""
        X = np.array([0.0, 0.0])
        result = ackley(X)
        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_ackley_global_minimum_3d(self):
        """Test that global minimum is at origin in 3D."""
        X = np.array([0.0, 0.0, 0.0])
        result = ackley(X)
        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_ackley_global_minimum_5d(self):
        """Test that global minimum is at origin in 5D."""
        X = np.zeros(5)
        result = ackley(X)
        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_ackley_single_point_positive(self):
        """Test that Ackley function returns positive values away from origin."""
        X = np.array([1.0, 1.0])
        result = ackley(X)
        assert result[0] > 0

    def test_ackley_single_point_2d(self):
        """Test single point evaluation in 2D."""
        X = np.array([1.0, 1.0])
        result = ackley(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_ackley_single_dimension(self):
        """Test Ackley function with single dimension."""
        X = np.array([1.0])
        result = ackley(X)
        assert result.shape == (1,)
        assert result[0] > 0


class TestAckleyMultiplePoints:
    """Test Ackley function with multiple points."""

    def test_ackley_multiple_points_2d(self):
        """Test multiple points evaluation in 2D."""
        X = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]])
        result = ackley(X)
        assert result.shape == (3,)
        assert result[0] == pytest.approx(0.0, abs=1e-10)
        assert result[1] > 0
        assert result[2] > 0

    def test_ackley_multiple_points_3d(self):
        """Test multiple points evaluation in 3D."""
        X = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, -0.5, 0.0]])
        result = ackley(X)
        assert result.shape == (3,)
        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_ackley_batch_evaluation(self):
        """Test batch evaluation with many points."""
        n_points = 100
        X = np.random.randn(n_points, 2)
        result = ackley(X)
        assert result.shape == (n_points,)
        assert np.all(result >= 0)  # All values should be non-negative


class TestAckleySymmetry:
    """Test symmetry properties of Ackley function."""

    def test_ackley_symmetry_about_origin(self):
        """Test that Ackley is symmetric about the origin."""
        X1 = np.array([1.0, 2.0])
        X2 = np.array([-1.0, -2.0])
        result1 = ackley(X1)
        result2 = ackley(X2)
        assert result1[0] == pytest.approx(result2[0], rel=1e-10)

    def test_ackley_coordinate_permutation(self):
        """Test that permuting coordinates gives same result."""
        X1 = np.array([1.0, 2.0])
        X2 = np.array([2.0, 1.0])
        result1 = ackley(X1)
        result2 = ackley(X2)
        assert result1[0] == pytest.approx(result2[0], rel=1e-10)


class TestAckleyDimensions:
    """Test Ackley function with different dimensions."""

    def test_ackley_1d(self):
        """Test 1D Ackley function."""
        X = np.array([0.5])
        result = ackley(X)
        assert result.shape == (1,)

    def test_ackley_2d(self):
        """Test 2D Ackley function."""
        X = np.array([0.5, 1.0])
        result = ackley(X)
        assert result.shape == (1,)

    def test_ackley_high_dimensional(self):
        """Test high-dimensional Ackley function."""
        for dim in [5, 10, 20]:
            X = np.zeros(dim)
            result = ackley(X)
            assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_ackley_varying_dimensions(self):
        """Test Ackley with varying dimensions."""
        for dim in range(1, 11):
            X = np.ones(dim)
            result = ackley(X)
            assert result.shape == (1,)
            assert result[0] > 0


class TestAckleyInputFormats:
    """Test different input formats."""

    def test_ackley_list_input(self):
        """Test Ackley with list input."""
        X = [1.0, 2.0]
        result = ackley(X)
        assert isinstance(result, np.ndarray)

    def test_ackley_1d_array_input(self):
        """Test Ackley with 1D array input."""
        X = np.array([1.0, 2.0])
        result = ackley(X)
        assert result.shape == (1,)

    def test_ackley_2d_array_input(self):
        """Test Ackley with 2D array input."""
        X = np.array([[1.0, 2.0]])
        result = ackley(X)
        assert result.shape == (1,)


class TestAckleyBoundaryValues:
    """Test Ackley function at boundary values."""

    def test_ackley_at_typical_bounds(self):
        """Test Ackley at typical search domain boundaries."""
        # Typical domain: [-32.768, 32.768]
        X = np.array([[32.768, 32.768], [-32.768, -32.768], [32.768, -32.768]])
        result = ackley(X)
        assert result.shape == (3,)
        assert np.all(result > 0)

    def test_ackley_large_values(self):
        """Test Ackley with large input values."""
        X = np.array([100.0, 100.0])
        result = ackley(X)
        assert result.shape == (1,)
        assert np.isfinite(result[0])


class TestAckleyNumericalProperties:
    """Test numerical properties of Ackley function."""

    def test_ackley_non_negative(self):
        """Test that Ackley function is always non-negative."""
        np.random.seed(42)
        X = np.random.uniform(-30, 30, size=(100, 5))
        result = ackley(X)
        assert np.all(result >= 0)

    def test_ackley_continuity(self):
        """Test continuity of Ackley function."""
        X1 = np.array([1.0, 1.0])
        X2 = np.array([1.001, 1.001])
        result1 = ackley(X1)
        result2 = ackley(X2)
        # Small change in input should give small change in output
        assert abs(result1[0] - result2[0]) < 0.01

    def test_ackley_local_variations(self):
        """Test that Ackley has local variations due to multimodality."""
        # Ackley has local variations but generally increases with distance
        X_near = np.array([0.1, 0.1])
        X_far = np.array([10.0, 10.0])
        result_near = ackley(X_near)
        result_far = ackley(X_far)

        # Far from origin should generally have higher values
        assert result_far[0] > result_near[0]


class TestAckleyReturnTypes:
    """Test return types of Ackley function."""

    def test_ackley_returns_ndarray(self):
        """Test that Ackley returns numpy ndarray."""
        X = np.array([1.0, 2.0])
        result = ackley(X)
        assert isinstance(result, np.ndarray)

    def test_ackley_returns_float_dtype(self):
        """Test that Ackley returns float dtype."""
        X = np.array([1.0, 2.0])
        result = ackley(X)
        assert np.issubdtype(result.dtype, np.floating)


class TestAckleyEdgeCases:
    """Test edge cases for Ackley function."""

    def test_ackley_very_small_values(self):
        """Test Ackley with very small values near origin."""
        X = np.array([1e-10, 1e-10])
        result = ackley(X)
        assert result[0] == pytest.approx(0.0, abs=1e-8)

    def test_ackley_mixed_signs(self):
        """Test Ackley with mixed positive and negative values."""
        X = np.array([[1.0, -1.0], [-2.0, 2.0], [3.0, -3.0]])
        result = ackley(X)
        assert result.shape == (3,)
        assert np.all(result > 0)


class TestAckleyOptimization:
    """Test Ackley function in optimization context."""

    def test_ackley_with_spotoptim(self):
        """Test that Ackley can be used with SpotOptim."""
        from spotoptim import SpotOptim

        opt = SpotOptim(
            fun=ackley, bounds=[(-5, 5), (-5, 5)], n_initial=5, max_iter=10, seed=42
        )

        result = opt.optimize()

        # Should find a solution close to origin
        assert result.fun < 5.0  # Should be much better than random
        assert np.all(np.abs(result.x) < 5.0)

    def test_ackley_gradient_descent_direction(self):
        """Test that moving toward origin decreases function value."""
        X1 = np.array([5.0, 5.0])
        X2 = np.array([2.5, 2.5])
        X3 = np.array([1.0, 1.0])

        result1 = ackley(X1)
        result2 = ackley(X2)
        result3 = ackley(X3)

        # Moving toward origin should decrease value
        assert result1[0] > result2[0]
        assert result2[0] > result3[0]


class TestAckleyDocstring:
    """Test examples from the docstring."""

    def test_docstring_example_global_minimum(self):
        """Test the global minimum example from docstring."""
        X = np.array([0.0, 0.0, 0.0])
        result = ackley(X)
        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_docstring_example_multiple_points(self):
        """Test the multiple points example from docstring."""
        X = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]])
        result = ackley(X)
        assert result[0] == pytest.approx(0.0, abs=1e-10)
        assert result[1] > 0
        assert result[2] > 0
