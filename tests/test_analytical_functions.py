"""Tests for analytical benchmark functions."""

import pytest
import numpy as np
from spotoptim.function import rosenbrock


class TestRosenbrock:
    """Test suite for the Rosenbrock function."""

    def test_global_minimum_2d(self):
        """Test that the global minimum is correctly identified at [1, 1]."""
        X = np.array([[1.0, 1.0]])
        result = rosenbrock(X)
        assert np.isclose(result[0], 0.0, atol=1e-10), (
            f"Global minimum should be 0, got {result[0]}"
        )

    def test_single_point_2d(self):
        """Test evaluation at a single 2D point."""
        X = np.array([0.0, 0.0])
        result = rosenbrock(X)
        expected = (1 - 0)**2 + 100 * (0 - 0**2)**2
        assert np.isclose(result[0], expected), (
            f"Expected {expected}, got {result[0]}"
        )

    def test_multiple_points_2d(self):
        """Test batch evaluation of multiple 2D points."""
        X = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5]
        ])
        result = rosenbrock(X)
        
        assert len(result) == 3, f"Expected 3 results, got {len(result)}"
        
        # Check first point [0, 0]
        expected_0 = (1 - 0)**2 + 100 * (0 - 0**2)**2
        assert np.isclose(result[0], expected_0), (
            f"Point [0, 0]: expected {expected_0}, got {result[0]}"
        )
        
        # Check second point [1, 1] (global minimum)
        assert np.isclose(result[1], 0.0, atol=1e-10), (
            f"Point [1, 1]: expected 0, got {result[1]}"
        )
        
        # Check third point [0.5, 0.5]
        expected_2 = (1 - 0.5)**2 + 100 * (0.5 - 0.5**2)**2
        assert np.isclose(result[2], expected_2), (
            f"Point [0.5, 0.5]: expected {expected_2}, got {result[2]}"
        )

    def test_known_values_2d(self):
        """Test against known function values."""
        test_cases = [
            # (x, y, expected_value)
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 4.0),
            (2.0, 4.0, 1.0),
        ]
        
        for x, y, expected in test_cases:
            X = np.array([[x, y]])
            result = rosenbrock(X)
            assert np.isclose(result[0], expected, atol=1e-10), (
                f"For point ({x}, {y}): expected {expected}, got {result[0]}"
            )

    def test_negative_values(self):
        """Test that function handles negative coordinates correctly."""
        X = np.array([[-1.0, -1.0]])
        result = rosenbrock(X)
        expected = (1 - (-1))**2 + 100 * ((-1) - (-1)**2)**2
        assert np.isclose(result[0], expected), (
            f"Expected {expected}, got {result[0]}"
        )

    def test_symmetry_breaking(self):
        """Test that function is not symmetric (as expected for Rosenbrock)."""
        X1 = np.array([[0.5, 0.3]])
        X2 = np.array([[0.3, 0.5]])
        
        result1 = rosenbrock(X1)
        result2 = rosenbrock(X2)
        
        # Rosenbrock is not symmetric, so these should differ
        assert not np.isclose(result1[0], result2[0]), (
            "Rosenbrock function should not be symmetric"
        )

    def test_input_shape_1d(self):
        """Test that 1D input is correctly converted to 2D."""
        X_1d = np.array([1.0, 1.0])
        result = rosenbrock(X_1d)
        assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"
        assert np.isclose(result[0], 0.0, atol=1e-10)

    def test_input_shape_2d(self):
        """Test that 2D input is handled correctly."""
        X_2d = np.array([[1.0, 1.0]])
        result = rosenbrock(X_2d)
        assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"

    def test_vectorized_computation(self):
        """Test that vectorized computation matches individual evaluations."""
        points = np.array([
            [0.0, 0.0],
            [0.5, 0.25],
            [1.0, 1.0],
            [-0.5, 0.25]
        ])
        
        # Batch evaluation
        batch_result = rosenbrock(points)
        
        # Individual evaluations
        individual_results = [rosenbrock(point.reshape(1, -1))[0] for point in points]
        
        np.testing.assert_allclose(
            batch_result, 
            individual_results,
            err_msg="Batch evaluation should match individual evaluations"
        )

    def test_dimensions_error(self):
        """Test that function raises error for insufficient dimensions."""
        X = np.array([[1.0]])  # Only 1D
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            rosenbrock(X)

    def test_higher_dimensions_3d(self):
        """Test Rosenbrock function in 3D."""
        X = np.array([[1.0, 1.0, 1.0]])
        result = rosenbrock(X)
        assert np.isclose(result[0], 0.0, atol=1e-10), (
            "Global minimum at [1,1,1] should be 0"
        )

    def test_higher_dimensions_4d(self):
        """Test Rosenbrock function in 4D."""
        X = np.array([[1.0, 1.0, 1.0, 1.0]])
        result = rosenbrock(X)
        assert np.isclose(result[0], 0.0, atol=1e-10), (
            "Global minimum at [1,1,1,1] should be 0"
        )

    def test_higher_dimensions_non_optimum(self):
        """Test Rosenbrock function at non-optimum point in higher dimensions."""
        X = np.array([[0.0, 0.0, 0.0]])
        result = rosenbrock(X)
        # For [0,0,0]: sum of two terms
        # Term 1 (i=0): 100*(0-0^2)^2 + (1-0)^2 = 1
        # Term 2 (i=1): 100*(0-0^2)^2 + (1-0)^2 = 1
        expected = 2.0
        assert np.isclose(result[0], expected), (
            f"Expected {expected}, got {result[0]}"
        )

    def test_output_type(self):
        """Test that output is always a numpy array."""
        X = np.array([[1.0, 1.0]])
        result = rosenbrock(X)
        assert isinstance(result, np.ndarray), (
            f"Expected numpy array, got {type(result)}"
        )

    def test_output_dtype(self):
        """Test that output has float dtype."""
        X = np.array([[1, 1]], dtype=int)
        result = rosenbrock(X)
        assert np.issubdtype(result.dtype, np.floating), (
            f"Expected float dtype, got {result.dtype}"
        )

    def test_large_values(self):
        """Test function behavior with large input values."""
        X = np.array([[10.0, 100.0]])
        result = rosenbrock(X)
        # Should be a large positive value
        assert result[0] > 0, "Function value should be positive"
        assert np.isfinite(result[0]), "Function value should be finite"

    def test_negative_large_values(self):
        """Test function behavior with large negative input values."""
        X = np.array([[-10.0, -100.0]])
        result = rosenbrock(X)
        # Should be a very large positive value
        assert result[0] > 0, "Function value should be positive"
        assert np.isfinite(result[0]), "Function value should be finite"

    def test_mixed_batch_dimensions(self):
        """Test batch with different dimensionalities (all 2D)."""
        X = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [2.0, 4.0]
        ])
        result = rosenbrock(X)
        assert len(result) == 4, f"Expected 4 results, got {len(result)}"
        assert all(np.isfinite(result)), "All results should be finite"

    def test_atleast_2d_behavior(self):
        """Test that atleast_2d conversion works as expected."""
        # Test various input shapes
        inputs = [
            np.array([1.0, 1.0]),           # 1D
            np.array([[1.0, 1.0]]),         # 2D single row
            np.array([[1.0, 1.0], [0.0, 0.0]])  # 2D multiple rows
        ]
        
        for X in inputs:
            result = rosenbrock(X)
            assert isinstance(result, np.ndarray)
            assert result.ndim == 1, f"Expected 1D output, got {result.ndim}D"

    def test_consistency_2d_vs_nd(self):
        """Test that 2D optimized path gives same result as N-D path."""
        # For 2D, both paths should give identical results
        X = np.array([[0.5, 0.25], [1.0, 1.0], [-0.5, 0.5]])
        result = rosenbrock(X)
        
        # Verify against manual calculation
        expected = []
        for point in X:
            x, y = point
            expected.append((1 - x)**2 + 100 * (y - x**2)**2)
        
        np.testing.assert_allclose(
            result,
            expected,
            err_msg="2D optimized path should match manual calculation"
        )

    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_reproducibility(self, seed):
        """Test that function is deterministic (same input -> same output)."""
        np.random.seed(seed)
        X = np.random.randn(10, 2)
        
        result1 = rosenbrock(X.copy())
        result2 = rosenbrock(X.copy())
        
        np.testing.assert_array_equal(
            result1,
            result2,
            err_msg="Function should be deterministic"
        )

    def test_empty_array_handling(self):
        """Test behavior with empty array (edge case)."""
        X = np.array([]).reshape(0, 2)
        result = rosenbrock(X)
        assert len(result) == 0, "Empty input should produce empty output"

    def test_performance_large_batch(self):
        """Test that function can handle large batches efficiently."""
        n_points = 10000
        X = np.random.randn(n_points, 2)
        result = rosenbrock(X)
        
        assert len(result) == n_points
        assert all(np.isfinite(result))
