# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Michalewicz function."""

import numpy as np
import pytest
from spotoptim.function import michalewicz


class TestMichalewiczBasic:
    """Test basic Michalewicz function functionality."""

    def test_michalewicz_2d_minimum(self):
        """Test known minimum value for 2D Michalewicz."""
        # Known minimum at approximately (2.20, 1.57) with value ≈ -1.8013
        X = np.array([[2.20, 1.57]])
        result = michalewicz(X)
        assert result[0] == pytest.approx(-1.8013, abs=0.01)

    def test_michalewicz_single_point_2d(self):
        """Test single point evaluation in 2D."""
        X = np.array([1.0, 1.0])
        result = michalewicz(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_michalewicz_negative_values(self):
        """Test that Michalewicz function returns negative values."""
        X = np.array([1.5, 1.5])
        result = michalewicz(X)
        assert result[0] < 0

    def test_michalewicz_single_dimension(self):
        """Test Michalewicz function with single dimension."""
        X = np.array([1.5])
        result = michalewicz(X)
        assert result.shape == (1,)


class TestMichalewiczSteepness:
    """Test Michalewicz function with different steepness parameters."""

    def test_michalewicz_steepness_m5(self):
        """Test Michalewicz with steepness m=5."""
        X = np.array([1.5, 1.5])
        result = michalewicz(X, m=5)
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_michalewicz_steepness_m10(self):
        """Test Michalewicz with default steepness m=10."""
        X = np.array([1.5, 1.5])
        result = michalewicz(X, m=10)
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_michalewicz_steepness_m20(self):
        """Test Michalewicz with high steepness m=20."""
        X = np.array([1.5, 1.5])
        result = michalewicz(X, m=20)
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_michalewicz_steepness_comparison(self):
        """Test that steepness parameter m affects function values."""
        X = np.array([2.0, 2.0])
        result_m5 = michalewicz(X, m=5)
        result_m10 = michalewicz(X, m=10)
        result_m20 = michalewicz(X, m=20)
        
        # Higher m should give steeper valleys (more extreme values)
        # All should be negative and different
        assert result_m5[0] < 0
        assert result_m10[0] < 0
        assert result_m20[0] < 0
        assert len(set([result_m5[0], result_m10[0], result_m20[0]])) == 3


class TestMichalewiczMultiplePoints:
    """Test Michalewicz function with multiple points."""

    def test_michalewicz_multiple_points_2d(self):
        """Test multiple points evaluation in 2D."""
        X = np.array([[1.0, 1.0], [2.0, 1.5], [1.5, 2.0]])
        result = michalewicz(X)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_michalewicz_multiple_points_5d(self):
        """Test multiple points evaluation in 5D."""
        X = np.array([
            [1.5, 1.5, 1.5, 1.5, 1.5],
            [2.0, 2.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0, 1.0, 1.0]
        ])
        result = michalewicz(X)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_michalewicz_batch_evaluation(self):
        """Test batch evaluation with many points."""
        n_points = 100
        X = np.random.uniform(0, np.pi, size=(n_points, 2))
        result = michalewicz(X)
        assert result.shape == (n_points,)
        assert np.all(np.isfinite(result))


class TestMichalewiczDimensions:
    """Test Michalewicz function with different dimensions."""

    def test_michalewicz_1d(self):
        """Test 1D Michalewicz function."""
        X = np.array([1.5])
        result = michalewicz(X)
        assert result.shape == (1,)

    def test_michalewicz_2d(self):
        """Test 2D Michalewicz function."""
        X = np.array([1.5, 1.5])
        result = michalewicz(X)
        assert result.shape == (1,)

    def test_michalewicz_5d_minimum(self):
        """Test known minimum value for 5D Michalewicz."""
        # Known global minimum ≈ -4.687658 at specific location
        # At x=1.5 for all dimensions, value will be different
        X = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
        result = michalewicz(X)
        # Should be negative
        assert result[0] < 0

    def test_michalewicz_10d_minimum(self):
        """Test known minimum value for 10D Michalewicz."""
        # Known global minimum ≈ -9.66015 at specific location
        # At x=1.5 for all dimensions, value will be different
        X = np.ones(10) * 1.5
        result = michalewicz(X)
        # Should be negative and more negative than 5D at same point
        X_5d = np.ones(5) * 1.5
        result_5d = michalewicz(X_5d)
        assert result[0] < 0
        assert result[0] < result_5d[0]

    def test_michalewicz_varying_dimensions(self):
        """Test Michalewicz with varying dimensions."""
        for dim in range(1, 11):
            X = np.ones(dim) * 1.5
            result = michalewicz(X)
            assert result.shape == (1,)
            assert np.isfinite(result[0])


class TestMichalewiczInputFormats:
    """Test different input formats."""

    def test_michalewicz_list_input(self):
        """Test Michalewicz with list input."""
        X = [1.0, 1.5]
        result = michalewicz(X)
        assert isinstance(result, np.ndarray)

    def test_michalewicz_1d_array_input(self):
        """Test Michalewicz with 1D array input."""
        X = np.array([1.0, 1.5])
        result = michalewicz(X)
        assert result.shape == (1,)

    def test_michalewicz_2d_array_input(self):
        """Test Michalewicz with 2D array input."""
        X = np.array([[1.0, 1.5]])
        result = michalewicz(X)
        assert result.shape == (1,)


class TestMichalewiczBoundaryValues:
    """Test Michalewicz function at boundary values."""

    def test_michalewicz_at_lower_bound(self):
        """Test Michalewicz at lower boundary (0)."""
        X = np.array([[0.0, 0.0], [0.0, 1.5], [1.5, 0.0]])
        result = michalewicz(X)
        assert result.shape == (3,)
        # At x=0, sin(0)=0, so contribution is 0
        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_michalewicz_at_upper_bound(self):
        """Test Michalewicz at upper boundary (π)."""
        X = np.array([
            [np.pi, np.pi],
            [np.pi, 1.5],
            [1.5, np.pi]
        ])
        result = michalewicz(X)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_michalewicz_within_typical_bounds(self):
        """Test Michalewicz within typical domain [0, π]."""
        np.random.seed(42)
        X = np.random.uniform(0, np.pi, size=(50, 5))
        result = michalewicz(X)
        assert np.all(np.isfinite(result))


class TestMichalewiczNumericalProperties:
    """Test numerical properties of Michalewicz function."""

    def test_michalewicz_continuity(self):
        """Test continuity of Michalewicz function."""
        X1 = np.array([1.5, 1.5])
        X2 = np.array([1.501, 1.501])
        result1 = michalewicz(X1)
        result2 = michalewicz(X2)
        # Small change in input should give small change in output
        assert abs(result1[0] - result2[0]) < 0.1

    def test_michalewicz_multimodal(self):
        """Test that Michalewicz is multimodal with multiple local minima."""
        # Test different regions
        points = [
            [0.5, 0.5],
            [1.5, 1.5],
            [2.5, 1.5],
            [1.5, 2.5]
        ]
        results = [michalewicz(np.array([p]))[0] for p in points]
        
        # Should have variation indicating multiple modes
        assert max(results) - min(results) > 0.5


class TestMichalewiczReturnTypes:
    """Test return types of Michalewicz function."""

    def test_michalewicz_returns_ndarray(self):
        """Test that Michalewicz returns numpy ndarray."""
        X = np.array([1.0, 1.5])
        result = michalewicz(X)
        assert isinstance(result, np.ndarray)

    def test_michalewicz_returns_float_dtype(self):
        """Test that Michalewicz returns float dtype."""
        X = np.array([1.0, 1.5])
        result = michalewicz(X)
        assert np.issubdtype(result.dtype, np.floating)


class TestMichalewiczEdgeCases:
    """Test edge cases for Michalewicz function."""

    def test_michalewicz_zero_input(self):
        """Test Michalewicz with zero input."""
        X = np.array([0.0, 0.0])
        result = michalewicz(X)
        # sin(0) = 0, so result should be 0
        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_michalewicz_small_m(self):
        """Test Michalewicz with small steepness parameter."""
        X = np.array([1.5, 1.5])
        result = michalewicz(X, m=1)
        assert np.isfinite(result[0])

    def test_michalewicz_large_m(self):
        """Test Michalewicz with large steepness parameter."""
        X = np.array([1.5, 1.5])
        result = michalewicz(X, m=50)
        assert np.isfinite(result[0])


class TestMichalewiczOptimization:
    """Test Michalewicz function in optimization context."""

    def test_michalewicz_with_spotoptim_2d(self):
        """Test that Michalewicz can be used with SpotOptim in 2D."""
        from spotoptim import SpotOptim
        
        opt = SpotOptim(
            fun=michalewicz,
            bounds=[(0, np.pi), (0, np.pi)],
            n_initial=10,
            max_iter=20,
            seed=42
        )
        
        result = opt.optimize()
        
        # Should find a reasonably good solution
        # 2D global minimum is approximately -1.8013
        # With limited iterations, may not reach global minimum
        assert result.fun < -0.5  # Should be significantly negative
        assert np.all(result.x >= 0)
        assert np.all(result.x <= np.pi)

    def test_michalewicz_with_spotoptim_5d(self):
        """Test that Michalewicz can be used with SpotOptim in 5D."""
        from spotoptim import SpotOptim
        
        opt = SpotOptim(
            fun=michalewicz,
            bounds=[(0, np.pi)] * 5,
            n_initial=15,
            max_iter=30,
            seed=42,
            de_x0_prob=0.0, # do not use best point as starting point
        )
        
        result = opt.optimize()
        
        # Should find a reasonably good solution
        # 5D global minimum is approximately -4.687658
        assert result.fun < -2.0  # Should be better than -2
        assert np.all(result.x >= 0)
        assert np.all(result.x <= np.pi)


class TestMichalewiczDimensionDependence:
    """Test how Michalewicz function varies with dimension."""

    def test_michalewicz_deeper_with_dimension(self):
        """Test that minimum becomes more negative with higher dimensions."""
        results = []
        for dim in [2, 5, 10]:
            X = np.ones(dim) * 1.5
            result = michalewicz(X)
            results.append(result[0])
        
        # Higher dimensions should give more negative values
        assert results[0] > results[1] > results[2]

    def test_michalewicz_dimension_scaling(self):
        """Test that function scales appropriately with dimension."""
        X_2d = np.array([1.5, 1.5])
        X_4d = np.array([1.5, 1.5, 1.5, 1.5])
        
        result_2d = michalewicz(X_2d)
        result_4d = michalewicz(X_4d)
        
        # 4D should be more negative than 2D
        assert result_4d[0] < result_2d[0]


class TestMichalewiczDocstring:
    """Test examples from the docstring."""

    def test_docstring_example_2d(self):
        """Test the 2D example from docstring."""
        X = np.array([[2.20, 1.57]])
        result = michalewicz(X)
        assert result[0] == pytest.approx(-1.8013, abs=0.01)

    def test_docstring_example_multiple_points(self):
        """Test the multiple points example from docstring."""
        X = np.array([[1.0, 1.0], [2.0, 1.5], [1.5, 2.0]])
        result = michalewicz(X)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_docstring_example_different_m(self):
        """Test the different steepness example from docstring."""
        X = np.array([[2.0, 2.0]])
        result_m5 = michalewicz(X, m=5)
        result_m10 = michalewicz(X, m=10)
        result_m20 = michalewicz(X, m=20)
        
        # Different m values should give different results
        assert result_m5[0] != result_m10[0]
        assert result_m10[0] != result_m20[0]
        assert all(r < 0 for r in [result_m5[0], result_m10[0], result_m20[0]])


class TestMichalewiczMathematicalProperties:
    """Test mathematical properties of Michalewicz function."""

    def test_michalewicz_sine_behavior(self):
        """Test that function exhibits expected sine-based behavior."""
        # At π/2, sin(π/2) = 1, should contribute to negative value
        X = np.array([np.pi/2, np.pi/2])
        result = michalewicz(X)
        assert result[0] < 0

    def test_michalewicz_dimension_contribution(self):
        """Test that each dimension contributes to the sum."""
        # Single dimension
        X_1d = np.array([1.5])
        result_1d = michalewicz(X_1d)
        
        # Same value in all dimensions
        X_2d = np.array([1.5, 1.5])
        result_2d = michalewicz(X_2d)
        
        # 2D should be more negative (sum of contributions)
        assert result_2d[0] < result_1d[0]
