"""Tests for dimension reduction (to_red_dim and to_all_dim) in SpotOptim."""

import numpy as np
import pytest
from spotoptim import SpotOptim


class TestDimensionReduction:
    """Test suite for dimension reduction in SpotOptim.

    These tests verify that:
    1. Fixed dimensions are correctly identified
    2. Optimization works in reduced space
    3. Results are correctly expanded back to full space
    4. Variable names and types are handled correctly
    """

    def test_no_reduction_when_all_bounds_different(self):
        """Test that no reduction occurs when all bounds are different."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-3, 3), (-1, 1)],
            max_iter=3,
            n_initial=3,
            seed=42,
        )

        assert not opt.red_dim
        assert opt.n_dim == 3
        assert len(opt.lower) == 3
        assert len(opt.upper) == 3

    def test_reduction_with_one_fixed_dimension(self):
        """Test reduction when one dimension is fixed."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (2, 2), (-3, 3)],  # Middle dimension fixed at 2
            max_iter=3,
            n_initial=3,
            seed=42,
        )

        assert opt.red_dim
        assert opt.n_dim == 2  # Reduced to 2 dimensions
        assert len(opt.lower) == 2
        assert len(opt.upper) == 2
        assert np.array_equal(opt.ident, [False, True, False])

    def test_reduction_with_multiple_fixed_dimensions(self):
        """Test reduction when multiple dimensions are fixed."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (2, 2), (3, 3), (-1, 1)],
            max_iter=3,
            n_initial=3,
            seed=42,
        )

        assert opt.red_dim
        assert opt.n_dim == 2  # Only first and last dimensions vary
        assert np.array_equal(opt.ident, [False, True, True, False])

    def test_all_dimensions_fixed(self):
        """Test when all dimensions are fixed."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(2, 2), (3, 3), (4, 4)],
            max_iter=1,
            n_initial=1,
            seed=42,
        )

        assert opt.red_dim
        assert opt.n_dim == 0
        assert np.array_equal(opt.ident, [True, True, True])

    def test_to_all_dim_expands_correctly(self):
        """Test that to_all_dim correctly expands reduced points."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (2, 2), (-3, 3)],
            max_iter=3,
            n_initial=3,
            seed=42,
        )

        # Create points in reduced space (only dimensions 0 and 2)
        X_red = np.array([[1.0, 3.0], [2.0, 4.0], [-1.0, -2.0]])

        X_full = opt.to_all_dim(X_red)

        # Should have 3 samples, 3 dimensions
        assert X_full.shape == (3, 3)

        # Check middle dimension is always 2.0
        assert np.allclose(X_full[:, 1], 2.0)

        # Check other dimensions match reduced input
        assert np.allclose(X_full[:, 0], X_red[:, 0])
        assert np.allclose(X_full[:, 2], X_red[:, 1])

    def test_to_red_dim_reduces_correctly(self):
        """Test that to_red_dim correctly reduces full points."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (2, 2), (-3, 3)],
            max_iter=3,
            n_initial=3,
            seed=42,
        )

        # Create points in full space
        X_full = np.array([[1.0, 2.0, 3.0], [4.0, 2.0, 5.0], [-1.0, 2.0, -2.0]])

        X_red = opt.to_red_dim(X_full)

        # Should have 3 samples, 2 dimensions (removed middle)
        assert X_red.shape == (3, 2)

        # Check dimensions match (0 and 2 from original)
        assert np.allclose(X_red[:, 0], X_full[:, 0])
        assert np.allclose(X_red[:, 1], X_full[:, 2])

    def test_roundtrip_conversion(self):
        """Test that to_all_dim and to_red_dim are inverse operations."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (2, 2), (3, 3), (-1, 1)],
            max_iter=3,
            n_initial=3,
            seed=42,
        )

        # Start with reduced points
        X_red_original = np.array([[1.0, 3.0], [2.0, 4.0]])

        # Expand to full, then reduce back
        X_full = opt.to_all_dim(X_red_original)
        X_red_back = opt.to_red_dim(X_full)

        # Should get back original reduced points
        assert np.allclose(X_red_original, X_red_back)

    def test_optimization_with_fixed_dimensions(self):
        """Test that optimization works correctly with fixed dimensions."""

        def objective(X):
            # f(x0, x1, x2) = x0^2 + x1^2 + x2^2
            # With x1=2 fixed, minimum should be at x0=0, x2=0
            return np.sum(X**2, axis=1)

        opt = SpotOptim(
            fun=objective,
            bounds=[(-5, 5), (2, 2), (-5, 5)],  # x1 fixed at 2
            max_iter=10,
            n_initial=5,
            seed=42,
            verbose=False,
        )

        result = opt.optimize()

        # Result should be in full dimensions
        assert result.x.shape == (3,)

        # Fixed dimension should be 2.0
        assert np.isclose(result.x[1], 2.0)

        # Other dimensions should be close to 0
        assert np.abs(result.x[0]) < 0.5
        assert np.abs(result.x[2]) < 0.5

        # Function value should be close to 4.0 (from x1=2)
        assert result.fun >= 4.0
        assert result.fun < 4.5

    def test_var_name_reduction(self):
        """Test that variable names are correctly reduced."""
        custom_names = ["temp", "pressure", "flow", "volume"]
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (100, 100), (1, 10), (5, 5)],  # pressure and volume fixed
            var_name=custom_names,
            max_iter=3,
            n_initial=3,
            seed=42,
        )

        # Check reduced names only include varying dimensions
        assert opt.var_name == ["temp", "flow"]
        assert opt.all_var_name == custom_names

    def test_var_type_reduction(self):
        """Test that variable types are correctly reduced."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (2, 2), (0, 10), (3, 3)],
            var_type=["float", "int", "int", "factor"],
            max_iter=3,
            n_initial=3,
            seed=42,
        )

        # Check reduced types only include varying dimensions
        assert opt.var_type == ["float", "int"]
        assert opt.all_var_type == ["float", "int", "int", "factor"]

    def test_initial_design_with_fixed_dimensions(self):
        """Test that initial design is generated in reduced space."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (2, 2), (-3, 3)],
            max_iter=10,
            n_initial=10,
            seed=42,
        )

        X0 = opt._generate_initial_design()

        # Initial design should be in reduced space (2 dimensions)
        assert X0.shape == (10, 2)

    def test_provided_initial_design_full_dimensions(self):
        """Test optimization with user-provided initial design in full dimensions."""

        def objective(X):
            return np.sum(X**2, axis=1)

        # User provides points in full dimensions
        X_start_full = np.array([[1.0, 2.0, 3.0], [0.5, 2.0, 1.0], [-1.0, 2.0, -1.0]])

        opt = SpotOptim(
            fun=objective,
            bounds=[(-5, 5), (2, 2), (-5, 5)],  # x1 fixed at 2
            max_iter=5,
            n_initial=3,
            seed=42,
        )

        result = opt.optimize(X0=X_start_full)

        # Result should be in full dimensions
        assert result.x.shape == (3,)
        assert np.isclose(result.x[1], 2.0)

    def test_evaluation_expands_to_full_dimensions(self):
        """Test that function evaluation receives full-dimensional points."""
        evaluation_dims = []

        def objective(X):
            evaluation_dims.append(X.shape[1])
            return np.sum(X**2, axis=1)

        opt = SpotOptim(
            fun=objective,
            bounds=[(-5, 5), (2, 2), (-5, 5)],
            max_iter=3,
            n_initial=3,
            seed=42,
        )

        opt.optimize()

        # All evaluations should have received 3D points
        assert all(dim == 3 for dim in evaluation_dims)

    def test_result_X_in_full_dimensions(self):
        """Test that result.X contains all evaluated points in full dimensions."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (2, 2), (-5, 5)],
            max_iter=5,
            n_initial=3,
            seed=42,
        )

        result = opt.optimize()

        # Result X should be in full dimensions
        assert result.X.shape[1] == 3

        # All points in column 1 should be 2.0
        assert np.allclose(result.X[:, 1], 2.0)

    def test_no_reduction_preserves_behavior(self):
        """Test that optimization without reduction works as before."""

        def sphere(X):
            return np.sum(X**2, axis=1)

        opt_no_red = SpotOptim(
            fun=sphere, bounds=[(-5, 5), (-5, 5)], max_iter=10, n_initial=5, seed=42
        )

        result = opt_no_red.optimize()

        assert result.x.shape == (2,)
        assert result.X.shape[1] == 2
        assert not opt_no_red.red_dim

    def test_complex_fixed_pattern(self):
        """Test with a complex pattern of fixed and varying dimensions."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                (1, 1),  # Fixed
                (-5, 5),  # Varying
                (2, 2),  # Fixed
                (-3, 3),  # Varying
                (0, 0),  # Fixed
                (1, 10),  # Varying
            ],
            max_iter=3,
            n_initial=3,
            seed=42,
        )

        assert opt.red_dim
        assert opt.n_dim == 3  # Three varying dimensions
        assert np.array_equal(opt.ident, [True, False, True, False, True, False])

        # Test expansion
        X_red = np.array([[1.0, 2.0, 5.0]])
        X_full = opt.to_all_dim(X_red)

        assert X_full.shape == (1, 6)
        assert X_full[0, 0] == 1.0  # Fixed value
        assert X_full[0, 1] == 1.0  # From reduced
        assert X_full[0, 2] == 2.0  # Fixed value
        assert X_full[0, 3] == 2.0  # From reduced
        assert X_full[0, 4] == 0.0  # Fixed value
        assert X_full[0, 5] == 5.0  # From reduced

    def test_zero_tolerance_fixed_dimensions(self):
        """Test that dimensions are fixed with exact equality (zero tolerance)."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (2.0, 2.0), (-3, 3)],
            max_iter=3,
            n_initial=3,
            seed=42,
        )

        assert opt.red_dim
        assert opt.all_lower[1] == 2.0
        assert opt.all_upper[1] == 2.0
