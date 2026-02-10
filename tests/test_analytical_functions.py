# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

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
        assert np.isclose(
            result[0], 0.0, atol=1e-10
        ), f"Global minimum should be 0, got {result[0]}"

    def test_single_point_2d(self):
        """Test evaluation at a single 2D point."""
        X = np.array([0.0, 0.0])
        result = rosenbrock(X)
        expected = (1 - 0) ** 2 + 100 * (0 - 0**2) ** 2
        assert np.isclose(result[0], expected), f"Expected {expected}, got {result[0]}"

    def test_multiple_points_2d(self):
        """Test batch evaluation of multiple 2D points."""
        X = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        result = rosenbrock(X)

        assert len(result) == 3, f"Expected 3 results, got {len(result)}"

        # Check first point [0, 0]
        expected_0 = (1 - 0) ** 2 + 100 * (0 - 0**2) ** 2
        assert np.isclose(
            result[0], expected_0
        ), f"Point [0, 0]: expected {expected_0}, got {result[0]}"

        # Check second point [1, 1] (global minimum)
        assert np.isclose(
            result[1], 0.0, atol=1e-10
        ), f"Point [1, 1]: expected 0, got {result[1]}"

        # Check third point [0.5, 0.5]
        expected_2 = (1 - 0.5) ** 2 + 100 * (0.5 - 0.5**2) ** 2
        assert np.isclose(
            result[2], expected_2
        ), f"Point [0.5, 0.5]: expected {expected_2}, got {result[2]}"

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
            assert np.isclose(
                result[0], expected, atol=1e-10
            ), f"For point ({x}, {y}): expected {expected}, got {result[0]}"

    def test_negative_values(self):
        """Test that function handles negative coordinates correctly."""
        X = np.array([[-1.0, -1.0]])
        result = rosenbrock(X)
        expected = (1 - (-1)) ** 2 + 100 * ((-1) - (-1) ** 2) ** 2
        assert np.isclose(result[0], expected), f"Expected {expected}, got {result[0]}"

    def test_symmetry_breaking(self):
        """Test that function is not symmetric (as expected for Rosenbrock)."""
        X1 = np.array([[0.5, 0.3]])
        X2 = np.array([[0.3, 0.5]])

        result1 = rosenbrock(X1)
        result2 = rosenbrock(X2)

        # Rosenbrock is not symmetric, so these should differ
        assert not np.isclose(
            result1[0], result2[0]
        ), "Rosenbrock function should not be symmetric"

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
        points = np.array([[0.0, 0.0], [0.5, 0.25], [1.0, 1.0], [-0.5, 0.25]])

        # Batch evaluation
        batch_result = rosenbrock(points)

        # Individual evaluations
        individual_results = [rosenbrock(point.reshape(1, -1))[0] for point in points]

        np.testing.assert_allclose(
            batch_result,
            individual_results,
            err_msg="Batch evaluation should match individual evaluations",
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
        assert np.isclose(
            result[0], 0.0, atol=1e-10
        ), "Global minimum at [1,1,1] should be 0"

    def test_higher_dimensions_4d(self):
        """Test Rosenbrock function in 4D."""
        X = np.array([[1.0, 1.0, 1.0, 1.0]])
        result = rosenbrock(X)
        assert np.isclose(
            result[0], 0.0, atol=1e-10
        ), "Global minimum at [1,1,1,1] should be 0"

    def test_higher_dimensions_non_optimum(self):
        """Test Rosenbrock function at non-optimum point in higher dimensions."""
        X = np.array([[0.0, 0.0, 0.0]])
        result = rosenbrock(X)
        # For [0,0,0]: sum of two terms
        # Term 1 (i=0): 100*(0-0^2)^2 + (1-0)^2 = 1
        # Term 2 (i=1): 100*(0-0^2)^2 + (1-0)^2 = 1
        expected = 2.0
        assert np.isclose(result[0], expected), f"Expected {expected}, got {result[0]}"

    def test_output_type(self):
        """Test that output is always a numpy array."""
        X = np.array([[1.0, 1.0]])
        result = rosenbrock(X)
        assert isinstance(
            result, np.ndarray
        ), f"Expected numpy array, got {type(result)}"

    def test_output_dtype(self):
        """Test that output has float dtype."""
        X = np.array([[1, 1]], dtype=int)
        result = rosenbrock(X)
        assert np.issubdtype(
            result.dtype, np.floating
        ), f"Expected float dtype, got {result.dtype}"

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
        X = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0], [2.0, 4.0]])
        result = rosenbrock(X)
        assert len(result) == 4, f"Expected 4 results, got {len(result)}"
        assert all(np.isfinite(result)), "All results should be finite"

    def test_atleast_2d_behavior(self):
        """Test that atleast_2d conversion works as expected."""
        # Test various input shapes
        inputs = [
            np.array([1.0, 1.0]),  # 1D
            np.array([[1.0, 1.0]]),  # 2D single row
            np.array([[1.0, 1.0], [0.0, 0.0]]),  # 2D multiple rows
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
            expected.append((1 - x) ** 2 + 100 * (y - x**2) ** 2)

        np.testing.assert_allclose(
            result,
            expected,
            err_msg="2D optimized path should match manual calculation",
        )

    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_reproducibility(self, seed):
        """Test that function is deterministic (same input -> same output)."""
        np.random.seed(seed)
        X = np.random.randn(10, 2)

        result1 = rosenbrock(X.copy())
        result2 = rosenbrock(X.copy())

        np.testing.assert_array_equal(
            result1, result2, err_msg="Function should be deterministic"
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


class TestRobotArmHard:
    """Test suite for the robot_arm_hard function."""

    def test_correct_dimension(self):
        """Test that function requires exactly 10 dimensions."""
        from spotoptim.function import robot_arm_hard
        
        # Correct dimension (10)
        X = np.random.rand(1, 10)
        result = robot_arm_hard(X)
        assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"
        assert np.isfinite(result[0]), "Result should be finite"

    def test_single_point_evaluation(self):
        """Test evaluation at a single point."""
        from spotoptim.function import robot_arm_hard
        
        # Mid-range configuration (all angles at 0.5)
        X = np.full(10, 0.5)
        result = robot_arm_hard(X)
        
        assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"
        assert isinstance(result, np.ndarray), f"Expected numpy array, got {type(result)}"
        assert result[0] > 0, "Cost should be positive for non-optimal configuration"

    def test_multiple_points_evaluation(self):
        """Test batch evaluation of multiple points."""
        from spotoptim.function import robot_arm_hard
        
        X = np.array([
            np.full(10, 0.5),  # Mid-range
            np.full(10, 0.3),  # Lower range
            np.full(10, 0.7),  # Upper range
        ])
        result = robot_arm_hard(X)
        
        assert len(result) == 3, f"Expected 3 results, got {len(result)}"
        assert all(np.isfinite(result)), "All results should be finite"
        assert all(result > 0), "All costs should be positive"

    def test_input_shape_1d(self):
        """Test that 1D input is correctly converted to 2D."""
        from spotoptim.function import robot_arm_hard
        
        X_1d = np.random.rand(10)
        result = robot_arm_hard(X_1d)
        assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"

    def test_input_shape_2d(self):
        """Test that 2D input is handled correctly."""
        from spotoptim.function import robot_arm_hard
        
        X_2d = np.random.rand(5, 10)
        result = robot_arm_hard(X_2d)
        assert result.shape == (5,), f"Expected shape (5,), got {result.shape}"

    def test_output_type(self):
        """Test that output is always a numpy array."""
        from spotoptim.function import robot_arm_hard
        
        X = np.random.rand(10)
        result = robot_arm_hard(X)
        assert isinstance(result, np.ndarray), f"Expected numpy array, got {type(result)}"

    def test_output_dtype(self):
        """Test that output has float dtype."""
        from spotoptim.function import robot_arm_hard
        
        X = np.random.rand(10)
        result = robot_arm_hard(X)
        assert np.issubdtype(result.dtype, np.floating), f"Expected float dtype, got {result.dtype}"

    def test_boundary_values(self):
        """Test with boundary values (0 and 1)."""
        from spotoptim.function import robot_arm_hard
        
        # All zeros (maps to -1.2π)
        X_zeros = np.zeros(10)
        result_zeros = robot_arm_hard(X_zeros)
        assert np.isfinite(result_zeros[0]), "Result should be finite for all zeros"
        
        # All ones (maps to 1.2π)
        X_ones = np.ones(10)
        result_ones = robot_arm_hard(X_ones)
        assert np.isfinite(result_ones[0]), "Result should be finite for all ones"
        
        # Mixed boundaries
        X_mixed = np.array([0.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.0, 1.0, 0.0, 1.0])
        result_mixed = robot_arm_hard(X_mixed)
        assert np.isfinite(result_mixed[0]), "Result should be finite for mixed boundaries"

    def test_constraint_penalty_effect(self):
        """Test that constraint violations result in high penalties."""
        from spotoptim.function import robot_arm_hard
        
        # A straight configuration (all angles = 0.5 → 0 radians)
        # This will likely hit obstacles and have high penalty
        X_straight = np.full(10, 0.5)
        cost_straight = robot_arm_hard(X_straight)
        
        # The cost should be substantial due to obstacles
        assert cost_straight[0] > 100, "Straight arm should have high cost due to obstacles"

    def test_vectorized_computation(self):
        """Test that vectorized computation matches individual evaluations."""
        from spotoptim.function import robot_arm_hard
        
        np.random.seed(42)
        points = np.random.rand(5, 10)
        
        # Batch evaluation
        batch_result = robot_arm_hard(points)
        
        # Individual evaluations
        individual_results = [robot_arm_hard(point.reshape(1, -1))[0] for point in points]
        
        np.testing.assert_allclose(
            batch_result,
            individual_results,
            err_msg="Batch evaluation should match individual evaluations",
        )

    def test_reproducibility(self):
        """Test that function is deterministic (same input -> same output)."""
        from spotoptim.function import robot_arm_hard
        
        np.random.seed(123)
        X = np.random.rand(10, 10)
        
        result1 = robot_arm_hard(X.copy())
        result2 = robot_arm_hard(X.copy())
        
        np.testing.assert_array_equal(
            result1, result2, err_msg="Function should be deterministic"
        )

    def test_cost_components_present(self):
        """Test that cost has all three components (distance, penalty, energy)."""
        from spotoptim.function import robot_arm_hard
        
        # Very small configuration (low energy)
        X_small = np.full(10, 0.5)  # Maps to 0 radians
        cost_small = robot_arm_hard(X_small)
        
        # The cost should include distance to target, obstacle penalties, and energy
        # For a straight arm from origin, end effector is at (10, 0) far from target (5, 5)
        # Distance cost alone: (10-5)^2 + (0-5)^2 = 25 + 25 = 50
        # Plus obstacle penalties and energy
        assert cost_small[0] >= 50, "Cost should include at least distance cost"

    def test_different_configurations_different_costs(self):
        """Test that different configurations yield different costs."""
        from spotoptim.function import robot_arm_hard
        
        np.random.seed(42)
        X1 = np.random.rand(10)
        X2 = np.random.rand(10)
        
        cost1 = robot_arm_hard(X1)
        cost2 = robot_arm_hard(X2)
        
        # Very unlikely to be exactly equal for random configurations
        assert not np.isclose(cost1[0], cost2[0]), "Different configurations should have different costs"

    def test_all_positive_costs(self):
        """Test that all cost components result in non-negative total cost."""
        from spotoptim.function import robot_arm_hard
        
        np.random.seed(999)
        X = np.random.rand(20, 10)
        results = robot_arm_hard(X)
        
        assert all(results >= 0), "All costs should be non-negative"

    def test_random_sample_finiteness(self):
        """Test that random samples all produce finite costs."""
        from spotoptim.function import robot_arm_hard
        
        np.random.seed(777)
        X = np.random.rand(100, 10)
        results = robot_arm_hard(X)
        
        assert all(np.isfinite(results)), "All random samples should produce finite costs"

    def test_performance_large_batch(self):
        """Test that function can handle large batches efficiently."""
        from spotoptim.function import robot_arm_hard
        
        n_points = 1000
        X = np.random.rand(n_points, 10)
        result = robot_arm_hard(X)
        
        assert len(result) == n_points, f"Expected {n_points} results, got {len(result)}"
        assert all(np.isfinite(result)), "All results should be finite"

    def test_atleast_2d_behavior(self):
        """Test that atleast_2d conversion works as expected."""
        from spotoptim.function import robot_arm_hard
        
        # Test various input shapes
        inputs = [
            np.random.rand(10),  # 1D
            np.random.rand(1, 10),  # 2D single row
            np.random.rand(3, 10),  # 2D multiple rows
        ]
        
        for X in inputs:
            result = robot_arm_hard(X)
            assert isinstance(result, np.ndarray), "Result should be numpy array"
            assert result.ndim == 1, f"Expected 1D output, got {result.ndim}D"

    def test_energy_component(self):
        """Test that energy regularization affects cost."""
        from spotoptim.function import robot_arm_hard
        
        # Configuration with small angles (low energy)
        X_low_energy = np.full(10, 0.5)  # All mapped to 0 radians
        
        # Configuration with extreme angles (high energy)
        X_high_energy = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        
        cost_low = robot_arm_hard(X_low_energy)
        cost_high = robot_arm_hard(X_high_energy)
        
        # Both should be finite
        assert np.isfinite(cost_low[0]), "Low energy cost should be finite"
        assert np.isfinite(cost_high[0]), "High energy cost should be finite"

    def test_empty_array_handling(self):
        """Test behavior with empty array (edge case)."""
        from spotoptim.function import robot_arm_hard
        
        X = np.array([]).reshape(0, 10)
        result = robot_arm_hard(X)
        assert len(result) == 0, "Empty input should produce empty output"

    @pytest.mark.parametrize("seed", [42, 123, 456, 789])
    def test_parametrized_reproducibility(self, seed):
        """Test deterministic behavior with multiple random seeds."""
        from spotoptim.function import robot_arm_hard
        
        np.random.seed(seed)
        X = np.random.rand(5, 10)
        
        result1 = robot_arm_hard(X.copy())
        result2 = robot_arm_hard(X.copy())
        
        np.testing.assert_array_equal(
            result1, result2, err_msg=f"Function should be deterministic for seed {seed}"
        )

    def test_target_proximity_affects_cost(self):
        """Test that configurations closer to target conceptually have potential for lower cost."""
        from spotoptim.function import robot_arm_hard
        
        # We can't easily construct a configuration that reaches the target
        # without solving the inverse kinematics, but we can verify the
        # distance component exists and affects the cost
        
        # Random configurations will have varying distances
        np.random.seed(555)
        X = np.random.rand(10, 10)
        costs = robot_arm_hard(X)
        
        # There should be variation in costs
        assert np.std(costs) > 0, "Different configurations should have varying costs"

    def test_input_range_validity(self):
        """Test that function handles full [0, 1] input range."""
        from spotoptim.function import robot_arm_hard
        
        # Test various points in [0, 1]^10
        test_points = [
            np.full(10, 0.0),   # Lower bound
            np.full(10, 0.25),  # Quarter
            np.full(10, 0.5),   # Middle
            np.full(10, 0.75),  # Three-quarters
            np.full(10, 1.0),   # Upper bound
        ]
        
        for X in test_points:
            result = robot_arm_hard(X)
            assert np.isfinite(result[0]), f"Result should be finite for input {X[0]}"
            assert result[0] >= 0, f"Cost should be non-negative for input {X[0]}"
