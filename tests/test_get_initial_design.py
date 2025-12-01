"""Tests for the get_initial_design() method in SpotOptim."""

import numpy as np
import pytest
from spotoptim import SpotOptim


class TestGetInitialDesignBasic:
    """Test basic get_initial_design() functionality."""

    def test_get_initial_design_default_lhs(self):
        """Test default behavior generates LHS design."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=10,
            seed=42,
        )
        
        X0 = opt.get_initial_design()
        
        # Check shape
        assert X0.shape == (10, 2)
        
        # Check bounds (internal scale)
        assert np.all(X0 >= opt.lower)
        assert np.all(X0 <= opt.upper)

    def test_get_initial_design_single_dimension(self):
        """Test get_initial_design() with single dimension."""
        opt = SpotOptim(
            fun=lambda X: X.flatten()**2,
            bounds=[(-10, 10)],
            n_initial=5,
            seed=42,
        )
        
        X0 = opt.get_initial_design()
        
        assert X0.shape == (5, 1)
        assert np.all(X0 >= -10)
        assert np.all(X0 <= 10)

    def test_get_initial_design_many_dimensions(self):
        """Test get_initial_design() with many dimensions."""
        n_dims = 10
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 1) for _ in range(n_dims)],
            n_initial=20,
            seed=42,
        )
        
        X0 = opt.get_initial_design()
        
        assert X0.shape == (20, n_dims)
        assert np.all(X0 >= 0)
        assert np.all(X0 <= 1)

    def test_get_initial_design_different_n_initial(self):
        """Test get_initial_design() with different n_initial values."""
        for n_init in [5, 10, 20, 50]:
            opt = SpotOptim(
                fun=lambda X: np.sum(X**2, axis=1),
                bounds=[(-5, 5), (-5, 5)],
                n_initial=n_init,
                max_iter=max(50, n_init),  # Ensure max_iter >= n_initial
                seed=42,
            )
            
            X0 = opt.get_initial_design()
            assert X0.shape == (n_init, 2)


class TestGetInitialDesignCustomX0:
    """Test get_initial_design() with custom X0."""

    def test_get_initial_design_with_custom_x0(self):
        """Test providing custom initial design."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=10,
            seed=42,
        )
        
        X0_custom = np.array([[0, 0], [1, 1], [2, 2]])
        X0 = opt.get_initial_design(X0_custom)
        
        # Should return 3 points (custom design)
        assert X0.shape == (3, 2)

    def test_get_initial_design_custom_x0_single_point(self):
        """Test providing single point as custom initial design."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=10,
            seed=42,
        )
        
        X0_custom = np.array([[1.5, 2.5]])
        X0 = opt.get_initial_design(X0_custom)
        
        assert X0.shape == (1, 2)

    def test_get_initial_design_custom_x0_transformed(self):
        """Test that custom X0 is transformed to internal scale."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1, 100)],
            var_trans=['log10'],
            n_initial=10,
            seed=42,
        )
        
        # Provide X0 in original scale (use floats to avoid numpy type issues)
        X0_custom = np.array([[10.0], [50.0], [100.0]])
        X0 = opt.get_initial_design(X0_custom)
        
        # Should be transformed to log10 scale
        assert X0.shape == (3, 1)
        # log10(10) = 1, log10(100) = 2
        assert X0[0, 0] == pytest.approx(np.log10(10))
        assert X0[2, 0] == pytest.approx(np.log10(100))

    def test_get_initial_design_custom_x0_with_int_var_type(self):
        """Test that custom X0 is rounded for integer variables."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10), (0, 10)],
            var_type=['int', 'float'],
            n_initial=10,
            seed=42,
        )
        
        X0_custom = np.array([[1.5, 2.5], [3.7, 4.2]])
        X0 = opt.get_initial_design(X0_custom)
        
        # First column should be rounded
        assert X0[0, 0] == 2.0
        assert X0[1, 0] == 4.0
        # Second column should remain float
        assert X0[0, 1] == pytest.approx(2.5)
        assert X0[1, 1] == pytest.approx(4.2)


class TestGetInitialDesignWithStartingPoint:
    """Test get_initial_design() with starting point x0."""

    def test_get_initial_design_includes_x0(self):
        """Test that x0 is included as first point in initial design."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=10,
            x0=np.array([1.0, 2.0]),
            seed=42,
        )
        
        X0 = opt.get_initial_design()
        
        # Should have n_initial points
        assert X0.shape == (10, 2)
        
        # First point should be x0 (in internal scale)
        assert X0[0, 0] == pytest.approx(1.0)
        assert X0[0, 1] == pytest.approx(2.0)

    def test_get_initial_design_x0_with_transformation(self):
        """Test that x0 is correctly included when transformations are used."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1, 100)],
            var_trans=['log10'],
            n_initial=5,
            x0=np.array([10.0]),
            seed=42,
        )
        
        X0 = opt.get_initial_design()
        
        assert X0.shape == (5, 1)
        # First point should be x0 in internal scale (log10(10) = 1)
        assert X0[0, 0] == pytest.approx(1.0)

    def test_get_initial_design_x0_not_included_when_x0_provided(self):
        """Test that x0 is not used when X0 is explicitly provided."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=10,
            x0=np.array([1.0, 2.0]),  # This should be ignored
            seed=42,
        )
        
        X0_custom = np.array([[0, 0], [3, 3]])
        X0 = opt.get_initial_design(X0_custom)
        
        # Should use custom X0, not include x0
        assert X0.shape == (2, 2)
        # First point should be from custom X0, not x0
        assert X0[0, 0] == pytest.approx(0.0)
        assert X0[0, 1] == pytest.approx(0.0)


class TestGetInitialDesignDimensionReduction:
    """Test get_initial_design() with dimension reduction."""

    def test_get_initial_design_with_fixed_dimension(self):
        """Test get_initial_design() with fixed (constant) dimension."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (2, 2), (-5, 5)],  # Middle dimension fixed
            n_initial=10,
            seed=42,
        )
        
        X0 = opt.get_initial_design()
        
        # Should have reduced dimensions (fixed dimension removed)
        assert X0.shape == (10, 2)

    def test_get_initial_design_custom_x0_with_dimension_reduction(self):
        """Test custom X0 with dimension reduction."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (3, 3), (-5, 5)],
            n_initial=10,
            seed=42,
        )
        
        # Provide X0 in full dimensions
        X0_custom = np.array([[1, 3, 2], [0, 3, 1]])
        X0 = opt.get_initial_design(X0_custom)
        
        # Should be reduced to 2 dimensions
        assert X0.shape == (2, 2)
        # Middle dimension (fixed at 3) should be removed
        assert X0[0, 0] == pytest.approx(1.0)
        assert X0[0, 1] == pytest.approx(2.0)


class TestGetInitialDesignVarTypes:
    """Test get_initial_design() with different variable types."""

    def test_get_initial_design_int_var_type(self):
        """Test get_initial_design() with integer variables."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10), (0, 10)],
            var_type=['int', 'int'],
            n_initial=10,
            seed=42,
        )
        
        X0 = opt.get_initial_design()
        
        # All values should be integers
        assert np.all(X0 == np.round(X0))

    def test_get_initial_design_mixed_var_types(self):
        """Test get_initial_design() with mixed variable types."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10), (0.0, 1.0), (5, 15)],
            var_type=['int', 'float', 'int'],
            n_initial=10,
            seed=42,
        )
        
        X0 = opt.get_initial_design()
        
        # First and third columns should be integers
        assert np.all(X0[:, 0] == np.round(X0[:, 0]))
        assert np.all(X0[:, 2] == np.round(X0[:, 2]))

    def test_get_initial_design_factor_var_type(self):
        """Test get_initial_design() with factor variables."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[('red', 'green', 'blue'), (0, 10)],
            n_initial=10,
            seed=42,
        )
        
        X0 = opt.get_initial_design()
        
        # Factor variable should be integer indices
        assert np.all(X0[:, 0] == np.round(X0[:, 0]))
        assert np.all(X0[:, 0] >= 0)
        assert np.all(X0[:, 0] <= 2)


class TestGetInitialDesignTransformations:
    """Test get_initial_design() with transformations."""

    def test_get_initial_design_log10_transformation(self):
        """Test get_initial_design() with log10 transformation."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1, 100)],
            var_trans=['log10'],
            n_initial=10,
            seed=42,
        )
        
        X0 = opt.get_initial_design()
        
        # Should be in transformed (log10) scale
        assert X0.shape == (10, 1)
        # log10(1) = 0, log10(100) = 2
        assert np.all(X0 >= 0)
        assert np.all(X0 <= 2)

    def test_get_initial_design_sqrt_transformation(self):
        """Test get_initial_design() with sqrt transformation."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1, 100)],
            var_trans=['sqrt'],
            n_initial=10,
            seed=42,
        )
        
        X0 = opt.get_initial_design()
        
        # Should be in transformed (sqrt) scale
        # sqrt(1) = 1, sqrt(100) = 10
        assert np.all(X0 >= 1)
        assert np.all(X0 <= 10)

    def test_get_initial_design_mixed_transformations(self):
        """Test get_initial_design() with mixed transformations."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1, 100), (0, 10), (0.1, 10)],
            var_trans=['log10', None, 'sqrt'],
            n_initial=10,
            seed=42,
        )
        
        X0 = opt.get_initial_design()
        
        assert X0.shape == (10, 3)
        # First column: log10 transformed
        assert np.all(X0[:, 0] >= 0)
        assert np.all(X0[:, 0] <= 2)
        # Second column: no transformation
        assert np.all(X0[:, 1] >= 0)
        assert np.all(X0[:, 1] <= 10)


class TestGetInitialDesignReproducibility:
    """Test reproducibility of get_initial_design()."""

    def test_get_initial_design_with_seed(self):
        """Test that same seed produces same design."""
        opt1 = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=10,
            seed=42,
        )
        
        opt2 = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=10,
            seed=42,
        )
        
        X0_1 = opt1.get_initial_design()
        X0_2 = opt2.get_initial_design()
        
        np.testing.assert_array_almost_equal(X0_1, X0_2)

    def test_get_initial_design_different_seeds(self):
        """Test that different seeds produce different designs."""
        opt1 = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=10,
            seed=42,
        )
        
        opt2 = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=10,
            seed=123,
        )
        
        X0_1 = opt1.get_initial_design()
        X0_2 = opt2.get_initial_design()
        
        # Should be different
        assert not np.allclose(X0_1, X0_2)


class TestGetInitialDesignEdgeCases:
    """Test edge cases for get_initial_design()."""

    def test_get_initial_design_small_n_initial(self):
        """Test get_initial_design() with very small n_initial."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=1,
            seed=42,
        )
        
        X0 = opt.get_initial_design()
        
        assert X0.shape == (1, 2)

    def test_get_initial_design_large_n_initial(self):
        """Test get_initial_design() with large n_initial."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=100,
            max_iter=100,  # Ensure max_iter >= n_initial
            seed=42,
        )
        
        X0 = opt.get_initial_design()
        
        assert X0.shape == (100, 2)

    def test_get_initial_design_narrow_bounds(self):
        """Test get_initial_design() with very narrow bounds."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 0.001), (0, 0.001)],
            n_initial=10,
            seed=42,
        )
        
        X0 = opt.get_initial_design()
        
        assert X0.shape == (10, 2)
        assert np.all(X0 >= 0)
        assert np.all(X0 <= 0.001)

    def test_get_initial_design_asymmetric_bounds(self):
        """Test get_initial_design() with asymmetric bounds."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-100, 10), (0, 0.1)],
            n_initial=10,
            seed=42,
        )
        
        X0 = opt.get_initial_design()
        
        assert X0.shape == (10, 2)
        assert np.all(X0[:, 0] >= -100)
        assert np.all(X0[:, 0] <= 10)
        assert np.all(X0[:, 1] >= 0)
        assert np.all(X0[:, 1] <= 0.1)


class TestGetInitialDesignReturnTypes:
    """Test return types of get_initial_design()."""

    def test_get_initial_design_returns_ndarray(self):
        """Test that get_initial_design() returns numpy ndarray."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=10,
            seed=42,
        )
        
        X0 = opt.get_initial_design()
        
        assert isinstance(X0, np.ndarray)

    def test_get_initial_design_returns_2d_array(self):
        """Test that get_initial_design() returns 2D array."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5)],
            n_initial=10,
            seed=42,
        )
        
        X0 = opt.get_initial_design()
        
        assert X0.ndim == 2


class TestGetInitialDesignIntegration:
    """Test integration of get_initial_design() with optimization."""

    def test_get_initial_design_used_in_optimize(self):
        """Test that get_initial_design() is properly used in optimize()."""
        call_log = []
        
        def objective(X):
            call_log.extend(X.tolist())
            return np.sum(X**2, axis=1)
        
        opt = SpotOptim(
            fun=objective,
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            max_iter=5,
            seed=42,
        )
        
        # Get initial design
        X0 = opt.get_initial_design()
        
        # Run optimization
        result = opt.optimize()
        
        # Check that initial design points were evaluated
        assert len(call_log) >= 5

    def test_get_initial_design_with_custom_x0_in_optimize(self):
        """Test custom X0 in optimization."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=3,  # Set to match custom X0 size
            max_iter=10,  # Ensure max_iter >= n_initial
            seed=42,
        )
        
        # Provide custom initial design
        X0_custom = np.array([[0, 0], [1, 1], [2, 2]])
        
        # This should work in optimize() method
        result = opt.optimize(X0=X0_custom)
        
        # Should have evaluated at least the 3 custom points
        assert opt.counter >= 3


class TestGetInitialDesignDocstring:
    """Test examples from the docstring."""

    def test_docstring_example_default_lhs(self):
        """Test the default LHS example from docstring."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=10
        )
        # Generate default LHS design
        X0 = opt.get_initial_design()
        assert X0.shape == (10, 2)

    def test_docstring_example_custom_x0(self):
        """Test the custom X0 example from docstring."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=10
        )
        # Provide custom initial design
        X0_custom = np.array([[0, 0], [1, 1], [2, 2]])
        X0_processed = opt.get_initial_design(X0_custom)
        assert X0_processed.shape == (3, 2)


class TestGetInitialDesignVerbose:
    """Test verbose output of get_initial_design()."""

    def test_get_initial_design_verbose_with_x0(self, capsys):
        """Test that verbose mode prints message when x0 is included."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=10,
            x0=np.array([1.0, 2.0]),
            seed=42,
            verbose=True,
        )
        
        X0 = opt.get_initial_design()
        
        captured = capsys.readouterr()
        assert "starting point x0" in captured.out.lower()
