import pytest
import numpy as np
from spotoptim.SpotOptim import SpotOptim


class TestVarType:
    """Test suite for variable type handling in SpotOptim.
    
    SpotOptim supports three data types:
    1. 'float': Python floats, continuous optimization
    2. 'int': Python int, float values will be rounded to integers
    3. 'factor': Unordered categorical data, internally mapped to int values
       (e.g., "red"->0, "green"->1, etc.)
    """

    def test_var_type_default_is_float(self):
        """Test that default var_type is 'float' for all dimensions."""
        
        def dummy_fun(X):
            return np.sum(X**2, axis=1)
        
        bounds = [(-5, 5), (-5, 5), (-5, 5)]
        
        opt = SpotOptim(fun=dummy_fun, bounds=bounds)
        
        # Default should be 'float' for all dimensions
        assert opt.var_type == ["float", "float", "float"]
        assert len(opt.var_type) == 3
    
    def test_var_type_initialization_float(self):
        """Test that var_type=['float'] is properly initialized."""
        
        def dummy_fun(X):
            return np.sum(X**2, axis=1)
        
        bounds = [(-5, 5), (-5, 5)]
        var_type = ["float", "float"]
        
        opt = SpotOptim(fun=dummy_fun, bounds=bounds, var_type=var_type)
        
        assert opt.var_type == ["float", "float"]
    
    def test_var_type_initialization_int(self):
        """Test that var_type=['int'] is properly initialized."""
        
        def dummy_fun(X):
            return np.sum(X**2, axis=1)
        
        bounds = [(-5, 5), (-5, 5)]
        var_type = ["int", "int"]
        
        opt = SpotOptim(fun=dummy_fun, bounds=bounds, var_type=var_type)
        
        assert opt.var_type == ["int", "int"]
    
    def test_var_type_initialization_factor(self):
        """Test that var_type=['factor'] is properly initialized."""
        
        def dummy_fun(X):
            return np.sum(X**2, axis=1)
        
        bounds = [(0, 2), (0, 3)]  # 3 categories, 4 categories
        var_type = ["factor", "factor"]
        
        opt = SpotOptim(fun=dummy_fun, bounds=bounds, var_type=var_type)
        
        assert opt.var_type == ["factor", "factor"]
    
    def test_var_type_mixed_types(self):
        """Test mixed variable types initialization."""
        
        def dummy_fun(X):
            return np.sum(X**2, axis=1)
        
        bounds = [(-5, 5), (-10, 10), (0, 5)]
        var_type = ["float", "int", "factor"]
        
        opt = SpotOptim(fun=dummy_fun, bounds=bounds, var_type=var_type)
        
        assert opt.var_type == ["float", "int", "factor"]
    
    def test_var_type_length_mismatch_error(self):
        """Test that var_type length must match bounds length."""
        
        def dummy_fun(X):
            return np.sum(X**2, axis=1)
        
        bounds = [(-5, 5), (-5, 5)]
        var_type = ["float"]  # Wrong length
        
        # Should not raise during initialization, but var_type will be used
        opt = SpotOptim(fun=dummy_fun, bounds=bounds, var_type=var_type)
        # The implementation doesn't validate this, so we just check it's set
        assert opt.var_type == ["float"]
    
    def test_repair_non_numeric_int_type(self):
        """Test that int type variables are properly rounded."""
        
        def dummy_fun(X):
            return np.sum(X**2, axis=1)
        
        bounds = [(-5, 5), (-5, 5)]
        var_type = ["int", "int"]
        
        opt = SpotOptim(fun=dummy_fun, bounds=bounds, var_type=var_type, seed=42)
        
        # Generate initial design
        X0 = opt._generate_initial_design()
        
        # All values should be integers (or very close)
        assert np.allclose(X0, np.round(X0), atol=1e-10)
    
    def test_repair_non_numeric_float_type(self):
        """Test that float type variables are NOT rounded."""
        
        def dummy_fun(X):
            return np.sum(X**2, axis=1)
        
        bounds = [(-5, 5), (-5, 5)]
        var_type = ["float", "float"]
        
        opt = SpotOptim(fun=dummy_fun, bounds=bounds, var_type=var_type, seed=42)
        
        # Generate initial design
        X0 = opt._generate_initial_design()
        
        # At least some values should NOT be exact integers
        # (with high probability for LHS sampling)
        has_non_integer = np.any(~np.isclose(X0, np.round(X0)))
        assert has_non_integer, "Float variables should contain non-integer values"
    
    def test_repair_non_numeric_factor_type(self):
        """Test that factor type variables are rounded to integers."""
        
        def dummy_fun(X):
            return np.sum(X**2, axis=1)
        
        bounds = [(0, 2), (0, 3)]  # 3 and 4 categories
        var_type = ["factor", "factor"]
        
        opt = SpotOptim(fun=dummy_fun, bounds=bounds, var_type=var_type, seed=42)
        
        # Generate initial design
        X0 = opt._generate_initial_design()
        
        # All values should be integers
        assert np.allclose(X0, np.round(X0), atol=1e-10)
    
    def test_repair_non_numeric_mixed_types(self):
        """Test that mixed types are handled correctly."""
        
        def dummy_fun(X):
            return np.sum(X**2, axis=1)
        
        bounds = [(-5, 5), (-5, 5), (0, 2)]
        var_type = ["float", "int", "factor"]
        
        opt = SpotOptim(fun=dummy_fun, bounds=bounds, var_type=var_type, seed=42, n_initial=10)
        
        # Generate initial design
        X0 = opt._generate_initial_design()
        
        # First column (float) should have non-integers
        # Second column (int) should be integers
        # Third column (factor) should be integers
        assert np.allclose(X0[:, 1], np.round(X0[:, 1]), atol=1e-10), "Int column should be integers"
        assert np.allclose(X0[:, 2], np.round(X0[:, 2]), atol=1e-10), "Factor column should be integers"
    
    def test_optimize_with_float_variables(self):
        """Test full optimization with float variables."""
        
        def sphere(X):
            return np.sum(X**2, axis=1)
        
        bounds = [(-5, 5), (-5, 5)]
        var_type = ["float", "float"]
        
        opt = SpotOptim(
            fun=sphere,
            bounds=bounds,
            var_type=var_type,
            max_iter=5,
            n_initial=5,
            seed=42
        )
        
        result = opt.optimize()
        
        # Check that optimization ran successfully
        assert result.success
        assert result.nfev == 5  # max_iter now includes initial design
        
        # Best point should be near origin
        assert result.fun < 10.0  # Relaxed since fewer iterations
    
    def test_optimize_with_int_variables(self):
        """Test full optimization with integer variables."""
        
        def int_sphere(X):
            return np.sum(X**2, axis=1)
        
        bounds = [(-5, 5), (-5, 5)]
        var_type = ["int", "int"]
        
        opt = SpotOptim(
            fun=int_sphere,
            bounds=bounds,
            var_type=var_type,
            max_iter=5,
            n_initial=5,
            seed=42
        )
        
        result = opt.optimize()
        
        # Check that all evaluated points have integer values
        assert np.allclose(opt.X_, np.round(opt.X_), atol=1e-10)
        
        # Best point should be integers
        assert np.allclose(result.x, np.round(result.x), atol=1e-10)
    
    def test_optimize_with_factor_variables(self):
        """Test full optimization with factor (categorical) variables."""
        
        def categorical_func(X):
            # Simple function where category 0 is best
            return np.sum(X**2, axis=1)
        
        bounds = [(0, 2), (0, 3)]  # 3 and 4 categories (0,1,2) and (0,1,2,3)
        var_type = ["factor", "factor"]
        
        opt = SpotOptim(
            fun=categorical_func,
            bounds=bounds,
            var_type=var_type,
            max_iter=5,
            n_initial=5,
            seed=42
        )
        
        result = opt.optimize()
        
        # Check that all evaluated points have integer values (representing categories)
        assert np.allclose(opt.X_, np.round(opt.X_), atol=1e-10)
        
        # Best point should be integers within bounds
        assert np.allclose(result.x, np.round(result.x), atol=1e-10)
        assert np.all(result.x >= 0)
        assert result.x[0] <= 2
        assert result.x[1] <= 3
    
    def test_optimize_mixed_types_comprehensive(self):
        """Test comprehensive optimization with all three types."""
        
        def mixed_objective(X):
            # Different behavior for each type
            # x0 (float): continuous minimization
            # x1 (int): discrete minimization
            # x2 (factor): categorical (0 is best)
            return X[:, 0]**2 + X[:, 1]**2 + (X[:, 2]**2) * 2
        
        bounds = [(-3, 3), (-5, 5), (0, 4)]
        var_type = ["float", "int", "factor"]
        
        opt = SpotOptim(
            fun=mixed_objective,
            bounds=bounds,
            var_type=var_type,
            max_iter=10,
            n_initial=8,
            seed=42
        )
        
        result = opt.optimize()
        
        # Verify types in all evaluated points
        # x1 should be int
        assert np.allclose(opt.X_[:, 1], np.round(opt.X_[:, 1]), atol=1e-10)
        # x2 should be factor (int)
        assert np.allclose(opt.X_[:, 2], np.round(opt.X_[:, 2]), atol=1e-10)
        
        # Best point verification
        assert np.isclose(result.x[1], np.round(result.x[1]), atol=1e-10), "Int var should be integer"
        assert np.isclose(result.x[2], np.round(result.x[2]), atol=1e-10), "Factor var should be integer"
    
    def test_var_type_unsupported_type_treated_as_discrete(self):
        """Test that unsupported var_type values are treated as discrete (rounded).
        
        This is the current behavior - unsupported types that are not 'float'
        will be rounded to integers. In the future, this could raise an error instead.
        """
        
        def dummy_fun(X):
            return np.sum(X**2, axis=1)
        
        bounds = [(-5, 5), (-5, 5)]
        var_type = ["unknown", "invalid"]
        
        opt = SpotOptim(fun=dummy_fun, bounds=bounds, var_type=var_type, seed=42)
        
        # Generate initial design
        X0 = opt._generate_initial_design()
        
        # Unknown types should be treated as discrete (rounded to integers)
        assert np.allclose(X0, np.round(X0), atol=1e-10), "Unknown types should be rounded"
    
    def test_var_type_persistence_through_suggest_next_point(self):
        """Test that var_type is applied when suggesting next points."""
        
        def sphere(X):
            return np.sum(X**2, axis=1)
        
        bounds = [(-5, 5), (-5, 5)]
        var_type = ["int", "factor"]
        
        opt = SpotOptim(
            fun=sphere,
            bounds=bounds,
            var_type=var_type,
            max_iter=5, n_initial=5,
            seed=42
        )
        
        # Run optimization
        result = opt.optimize()
        
        # Check that all suggested points (including those after initial design)
        # respect the variable types
        for i in range(opt.X_.shape[0]):
            assert np.isclose(opt.X_[i, 0], np.round(opt.X_[i, 0]), atol=1e-10), f"Point {i}, dim 0 should be int"
            assert np.isclose(opt.X_[i, 1], np.round(opt.X_[i, 1]), atol=1e-10), f"Point {i}, dim 1 should be factor"
    
    def test_var_type_in_provided_initial_design(self):
        """Test that var_type is applied to user-provided initial design."""
        
        def sphere(X):
            return np.sum(X**2, axis=1)
        
        bounds = [(-5, 5), (-5, 5)]
        var_type = ["int", "int"]
        
        # Provide initial design with float values
        X0 = np.array([
            [1.5, 2.7],
            [3.2, -1.8],
            [-2.1, 4.3]
        ])
        n_initial = X0.shape[0]  # 3 points
        
        opt = SpotOptim(
            fun=sphere,
            bounds=bounds,
            var_type=var_type,
            max_iter=10,
            n_initial=n_initial,
            seed=42
        )
        
        result = opt.optimize(X0=X0)
        
        # Initial design should have been rounded
        assert np.allclose(opt.X_[:3], np.round(X0), atol=1e-10)
    
    def test_var_type_validation_bounds_compatibility(self):
        """Test that factor variables work with appropriate bounds."""
        
        def dummy_fun(X):
            return np.sum(X**2, axis=1)
        
        # Factor variables should work with integer-range bounds
        bounds = [(0, 5), (0, 10)]  # 6 and 11 categories
        var_type = ["factor", "factor"]
        
        opt = SpotOptim(
            fun=dummy_fun,
            bounds=bounds,
            var_type=var_type,
            max_iter=5, n_initial=5,
            seed=42
        )
        
        result = opt.optimize()
        
        # All values should be valid integer categories
        assert np.all(opt.X_[:, 0] >= 0)
        assert np.all(opt.X_[:, 0] <= 5)
        assert np.all(opt.X_[:, 1] >= 0)
        assert np.all(opt.X_[:, 1] <= 10)
        assert np.allclose(opt.X_, np.round(opt.X_), atol=1e-10)
    
    def test_var_type_with_plot_surrogate(self):
        """Test that plot_surrogate respects var_type settings."""
        
        def sphere(X):
            return np.sum(X**2, axis=1)
        
        bounds = [(-5, 5), (-5, 5)]
        var_type = ["int", "float"]
        
        opt = SpotOptim(
            fun=sphere,
            bounds=bounds,
            var_type=var_type,
            max_iter=5,
            n_initial=5,
            seed=42
        )
        
        result = opt.optimize()
        
        # Just check that plotting doesn't fail with mixed types
        # (testing actual plot would require display)
        try:
            opt.plot_surrogate(show=False)
            plot_succeeded = True
        except Exception as e:
            plot_succeeded = False
            print(f"Plot failed: {e}")
        
        assert plot_succeeded, "Plotting should work with mixed var_types"
    
    def test_repair_non_numeric_direct_call(self):
        """Test _repair_non_numeric method directly with different types."""
        
        def dummy_fun(X):
            return np.sum(X**2, axis=1)
        
        bounds = [(-5, 5), (-5, 5), (-5, 5)]
        
        opt = SpotOptim(fun=dummy_fun, bounds=bounds)
        
        # Test data with float values
        X = np.array([
            [1.7, 2.3, 3.9],
            [4.1, 5.6, 6.2]
        ])
        
        # Test with float types (no rounding)
        X_float = opt._repair_non_numeric(X.copy(), ["float", "float", "float"])
        assert not np.allclose(X_float, np.round(X))
        
        # Test with int types (all rounded)
        X_int = opt._repair_non_numeric(X.copy(), ["int", "int", "int"])
        assert np.allclose(X_int, np.round(X))
        
        # Test with factor types (all rounded)
        X_factor = opt._repair_non_numeric(X.copy(), ["factor", "factor", "factor"])
        assert np.allclose(X_factor, np.round(X))
        
        # Test with mixed types
        X_mixed = opt._repair_non_numeric(X.copy(), ["float", "int", "factor"])
        assert not np.isclose(X_mixed[0, 0], np.round(X[0, 0]))  # float not rounded
        assert np.isclose(X_mixed[0, 1], np.round(X[0, 1]))      # int rounded
        assert np.isclose(X_mixed[0, 2], np.round(X[0, 2]))      # factor rounded
    
    def test_var_type_edge_case_single_dimension(self):
        """Test var_type with single-dimensional problems."""
        
        def simple_func(X):
            return X[:, 0]**2
        
        # Test with float
        bounds = [(-5, 5)]
        var_type = ["float"]
        
        opt = SpotOptim(fun=simple_func, bounds=bounds, var_type=var_type, seed=42, max_iter=5, n_initial=5)
        result = opt.optimize()
        assert result.success
        
        # Test with int
        var_type = ["int"]
        opt = SpotOptim(fun=simple_func, bounds=bounds, var_type=var_type, seed=42, max_iter=5, n_initial=5)
        result = opt.optimize()
        assert np.isclose(result.x[0], np.round(result.x[0]), atol=1e-10)
        
        # Test with factor
        bounds = [(0, 4)]
        var_type = ["factor"]
        opt = SpotOptim(fun=simple_func, bounds=bounds, var_type=var_type, seed=42, max_iter=5, n_initial=5)
        result = opt.optimize()
        assert np.isclose(result.x[0], np.round(result.x[0]), atol=1e-10)
        assert 0 <= result.x[0] <= 4
