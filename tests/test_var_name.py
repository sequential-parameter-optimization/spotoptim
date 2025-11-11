"""Tests for variable names (var_name) feature in SpotOptim."""

import numpy as np
import pytest
from spotoptim import SpotOptim


class TestVarName:
    """Test suite for variable name handling in SpotOptim.
    
    These tests verify that:
    1. Default variable names are correctly assigned
    2. Custom variable names can be specified
    3. Variable names are used in plotting
    """

    def test_var_name_default(self):
        """Test that default variable names are x0, x1, x2, ..."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5), (-5, 5)],
            max_iter=3, n_initial=3,
            seed=42
        )
        
        assert opt.var_name == ["x0", "x1", "x2"]

    def test_var_name_custom(self):
        """Test that custom variable names are correctly assigned."""
        custom_names = ["temperature", "pressure", "flow_rate"]
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5), (-5, 5)],
            max_iter=3, n_initial=3,
            var_name=custom_names,
            seed=42
        )
        
        assert opt.var_name == custom_names

    def test_var_name_length_matches_dimensions(self):
        """Test that the number of variable names matches the number of dimensions."""
        n_dims = 5
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5)] * n_dims,
            max_iter=3, n_initial=3,
            seed=42
        )
        
        assert len(opt.var_name) == n_dims

    def test_var_name_with_different_var_types(self):
        """Test variable names work with different variable types."""
        custom_names = ["continuous", "discrete", "categorical"]
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (0, 10), (0, 3)],
            var_type=["float", "int", "factor"],
            var_name=custom_names,
            max_iter=3, n_initial=3,
            seed=42
        )
        
        assert opt.var_name == custom_names
        assert opt.var_type == ["float", "int", "factor"]

    def test_var_name_single_dimension(self):
        """Test variable names for single dimension optimization."""
        opt = SpotOptim(
            fun=lambda X: X**2,
            bounds=[(-5, 5)],
            max_iter=3, n_initial=3,
            var_name=["x"],
            seed=42
        )
        
        assert opt.var_name == ["x"]

    def test_var_name_large_dimensions(self):
        """Test variable names for high-dimensional optimization."""
        n_dims = 20
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5)] * n_dims,
            max_iter=10, n_initial=10,
            seed=42
        )
        
        # Check that default names are x0, x1, ..., x19
        expected_names = [f"x{i}" for i in range(n_dims)]
        assert opt.var_name == expected_names

    def test_var_name_preserved_after_optimization(self):
        """Test that variable names are preserved after running optimization."""
        custom_names = ["alpha", "beta"]
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=3, n_initial=3,
            var_name=custom_names,
            seed=42
        )
        
        result = opt.optimize()
        
        # Verify names are preserved
        assert opt.var_name == custom_names
        assert result is not None

    def test_var_name_in_plot_surrogate_default(self):
        """Test that plot_surrogate uses instance var_name by default."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        custom_names = ["param1", "param2"]
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=5, n_initial=5,
            var_name=custom_names,
            seed=42
        )
        
        result = opt.optimize()
        
        # This should use instance var_name (param1, param2)
        # We can't easily test the plot labels directly without inspecting the figure,
        # but we can verify the method runs without error
        try:
            opt.plot_surrogate(i=0, j=1, show=False)
            success = True
        except Exception as e:
            success = False
            print(f"Plot failed with error: {e}")
        
        assert success, "plot_surrogate should run successfully with instance var_name"

    def test_var_name_in_plot_surrogate_override(self):
        """Test that plot_surrogate can override instance var_name."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        custom_names = ["param1", "param2"]
        override_names = ["override1", "override2"]
        
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=5, n_initial=5,
            var_name=custom_names,
            seed=42
        )
        
        result = opt.optimize()
        
        # This should use override names
        try:
            opt.plot_surrogate(i=0, j=1, var_name=override_names, show=False)
            success = True
        except Exception as e:
            success = False
            print(f"Plot failed with error: {e}")
        
        assert success, "plot_surrogate should run successfully with override var_name"
        # Instance var_name should remain unchanged
        assert opt.var_name == custom_names

    def test_var_name_with_special_characters(self):
        """Test that variable names can contain special characters."""
        special_names = ["x_1", "x-2", "x.3", "x(4)"]
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5)] * 4,
            max_iter=3, n_initial=3,
            var_name=special_names,
            seed=42
        )
        
        assert opt.var_name == special_names

    def test_var_name_with_unicode(self):
        """Test that variable names can contain unicode characters."""
        unicode_names = ["α", "β", "γ"]
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5)] * 3,
            max_iter=3, n_initial=3,
            var_name=unicode_names,
            seed=42
        )
        
        assert opt.var_name == unicode_names
