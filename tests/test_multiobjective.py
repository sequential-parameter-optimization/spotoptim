"""
Tests for multi-objective optimization support.
"""

import pytest
import numpy as np
from spotoptim import SpotOptim


class TestMultiObjectiveBasics:
    """Test basic multi-objective functionality."""

    def test_single_objective_function(self):
        """Test that single-objective functions work as before."""
        def single_obj(X):
            return np.sum(X**2, axis=1)
        
        optimizer = SpotOptim(
            fun=single_obj,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=15,
            n_initial=10,
            seed=42
        )
        
        result = optimizer.optimize()
        
        # Should have single-objective values
        assert result.success
        assert optimizer.y_mo is None  # No multi-objective storage
        assert optimizer.y_.ndim == 1
        assert len(optimizer.y_) == 15

    def test_multi_objective_function_default(self):
        """Test multi-objective function with default conversion (first objective)."""
        def multi_obj(X):
            # Return two objectives: sum of squares and sum of (x-1)^2
            obj1 = np.sum(X**2, axis=1)
            obj2 = np.sum((X - 1)**2, axis=1)
            return np.column_stack([obj1, obj2])
        
        optimizer = SpotOptim(
            fun=multi_obj,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=15,
            n_initial=10,
            seed=42
        )
        
        result = optimizer.optimize()
        
        # Should store multi-objective values
        assert result.success
        assert optimizer.y_mo is not None
        assert optimizer.y_mo.ndim == 2
        assert optimizer.y_mo.shape[1] == 2  # Two objectives
        assert optimizer.y_mo.shape[0] == 15  # All evaluations
        
        # y_ should be first objective
        np.testing.assert_array_equal(optimizer.y_, optimizer.y_mo[:, 0])

    def test_multi_objective_with_custom_conversion(self):
        """Test multi-objective function with custom conversion."""
        def multi_obj(X):
            obj1 = np.sum(X**2, axis=1)
            obj2 = np.sum((X - 1)**2, axis=1)
            return np.column_stack([obj1, obj2])
        
        def custom_mo2so(y_mo):
            # Weighted sum: 0.3*obj1 + 0.7*obj2
            return 0.3 * y_mo[:, 0] + 0.7 * y_mo[:, 1]
        
        optimizer = SpotOptim(
            fun=multi_obj,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=15,
            n_initial=10,
            fun_mo2so=custom_mo2so,
            seed=42
        )
        
        result = optimizer.optimize()
        
        # Should store multi-objective values
        assert optimizer.y_mo is not None
        
        # y_ should be the custom weighted sum
        expected_y = 0.3 * optimizer.y_mo[:, 0] + 0.7 * optimizer.y_mo[:, 1]
        np.testing.assert_array_almost_equal(optimizer.y_, expected_y)

    def test_get_shape_single_objective(self):
        """Test _get_shape for single-objective arrays."""
        optimizer = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5
        )
        
        y_single = np.array([1.0, 2.0, 3.0])
        n, m = optimizer._get_shape(y_single)
        
        assert n == 3
        assert m is None

    def test_get_shape_multi_objective(self):
        """Test _get_shape for multi-objective arrays."""
        optimizer = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5
        )
        
        y_multi = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        n, m = optimizer._get_shape(y_multi)
        
        assert n == 3
        assert m == 2


class TestMultiObjectiveStorage:
    """Test multi-objective storage functionality."""

    def test_store_mo_initial(self):
        """Test initial storage of multi-objective values."""
        optimizer = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5
        )
        
        y_mo = np.array([[1.0, 2.0], [3.0, 4.0]])
        optimizer._store_mo(y_mo)
        
        assert optimizer.y_mo is not None
        np.testing.assert_array_equal(optimizer.y_mo, y_mo)

    def test_store_mo_append(self):
        """Test appending multi-objective values."""
        optimizer = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5
        )
        
        y_mo_1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        optimizer._store_mo(y_mo_1)
        
        y_mo_2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        optimizer._store_mo(y_mo_2)
        
        expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        np.testing.assert_array_equal(optimizer.y_mo, expected)

    def test_store_mo_single_objective_ignored(self):
        """Test that single-objective arrays don't create y_mo."""
        optimizer = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5
        )
        
        y_single = np.array([1.0, 2.0, 3.0])
        optimizer._store_mo(y_single)
        
        assert optimizer.y_mo is None


class TestMultiObjectiveConversion:
    """Test multi-objective to single-objective conversion."""

    def test_mo2so_default_conversion(self):
        """Test default conversion uses first objective."""
        optimizer = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5
        )
        
        y_mo = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_so = optimizer._mo2so(y_mo)
        
        expected = np.array([1.0, 3.0, 5.0])
        np.testing.assert_array_equal(y_so, expected)

    def test_mo2so_custom_conversion(self):
        """Test custom conversion function."""
        def custom_conversion(y_mo):
            # Sum of all objectives
            return np.sum(y_mo, axis=1)
        
        optimizer = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5,
            fun_mo2so=custom_conversion
        )
        
        y_mo = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_so = optimizer._mo2so(y_mo)
        
        expected = np.array([3.0, 7.0, 11.0])
        np.testing.assert_array_equal(y_so, expected)

    def test_mo2so_single_objective_passthrough(self):
        """Test that single-objective arrays pass through unchanged."""
        optimizer = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5
        )
        
        y_single = np.array([1.0, 2.0, 3.0])
        y_so = optimizer._mo2so(y_single)
        
        np.testing.assert_array_equal(y_so, y_single)

    def test_mo2so_three_objectives(self):
        """Test conversion with three objectives."""
        def custom_conversion(y_mo):
            # Weighted sum of three objectives
            return 0.5 * y_mo[:, 0] + 0.3 * y_mo[:, 1] + 0.2 * y_mo[:, 2]
        
        optimizer = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5,
            fun_mo2so=custom_conversion
        )
        
        y_mo = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y_so = optimizer._mo2so(y_mo)
        
        expected = np.array([0.5*1 + 0.3*2 + 0.2*3, 0.5*4 + 0.3*5 + 0.2*6])
        np.testing.assert_array_almost_equal(y_so, expected)


class TestMultiObjectiveOptimization:
    """Test full optimization with multi-objective functions."""

    def test_optimization_multi_objective_sphere(self):
        """Test optimization of multi-objective sphere function."""
        def multi_sphere(X):
            # Two sphere functions with different centers
            obj1 = np.sum(X**2, axis=1)
            obj2 = np.sum((X - 2)**2, axis=1)
            return np.column_stack([obj1, obj2])
        
        optimizer = SpotOptim(
            fun=multi_sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=30,
            n_initial=15,
            seed=42,
            verbose=False
        )
        
        result = optimizer.optimize()
        
        assert result.success
        assert optimizer.y_mo is not None
        assert optimizer.y_mo.shape[0] == 30
        assert optimizer.y_mo.shape[1] == 2
        
        # Should optimize first objective (default)
        assert result.fun < 1.0  # Should find good solution

    def test_optimization_with_weighted_sum(self):
        """Test optimization with weighted sum scalarization."""
        def multi_obj(X):
            obj1 = np.sum(X**2, axis=1)
            obj2 = np.sum((X - 1)**2, axis=1)
            return np.column_stack([obj1, obj2])
        
        def weighted_sum(y_mo):
            return 0.5 * y_mo[:, 0] + 0.5 * y_mo[:, 1]
        
        optimizer = SpotOptim(
            fun=multi_obj,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=30,
            n_initial=15,
            fun_mo2so=weighted_sum,
            seed=42,
            verbose=False
        )
        
        result = optimizer.optimize()
        
        assert result.success
        # Should find compromise solution around x=0.5
        assert np.allclose(result.x, 0.5, atol=0.5)

    def test_optimization_with_noise_and_mo(self):
        """Test multi-objective optimization with noise."""
        def noisy_multi_obj(X):
            obj1 = np.sum(X**2, axis=1) + np.random.normal(0, 0.01, X.shape[0])
            obj2 = np.sum((X - 1)**2, axis=1) + np.random.normal(0, 0.01, X.shape[0])
            return np.column_stack([obj1, obj2])
        
        optimizer = SpotOptim(
            fun=noisy_multi_obj,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=30,
            n_initial=15,
            repeats_initial=2,
            repeats_surrogate=2,
            seed=42,
            verbose=False
        )
        
        result = optimizer.optimize()
        
        assert result.success
        assert optimizer.y_mo is not None
        assert optimizer.noise is True


class TestMultiObjectiveEdgeCases:
    """Test edge cases for multi-objective optimization."""

    def test_empty_array_handling(self):
        """Test handling of empty arrays."""
        optimizer = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5
        )
        
        y_empty = np.array([]).reshape(0, 2)
        y_so = optimizer._mo2so(y_empty)
        
        assert y_so.size == 0

    def test_single_sample_multi_objective(self):
        """Test multi-objective with single sample."""
        def multi_obj(X):
            return np.column_stack([
                np.sum(X**2, axis=1),
                np.sum((X-1)**2, axis=1)
            ])
        
        optimizer = SpotOptim(
            fun=multi_obj,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=12,
            n_initial=10,
            seed=42
        )
        
        result = optimizer.optimize()
        
        assert result.success
        assert optimizer.y_mo.shape[1] == 2

    def test_many_objectives(self):
        """Test with many objectives (5)."""
        def multi_obj(X):
            # Five different objectives
            objectives = []
            for i in range(5):
                objectives.append(np.sum((X - i)**2, axis=1))
            return np.column_stack(objectives)
        
        def sum_all(y_mo):
            return np.sum(y_mo, axis=1)
        
        optimizer = SpotOptim(
            fun=multi_obj,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=20,
            n_initial=12,
            fun_mo2so=sum_all,
            seed=42,
            verbose=False
        )
        
        result = optimizer.optimize()
        
        assert result.success
        assert optimizer.y_mo.shape[1] == 5


class TestMultiObjectiveWithFeatures:
    """Test multi-objective with other SpotOptim features."""

    def test_mo_with_dimension_reduction(self):
        """Test multi-objective with fixed dimensions."""
        def multi_obj(X):
            return np.column_stack([
                np.sum(X**2, axis=1),
                np.sum((X-1)**2, axis=1)
            ])
        
        optimizer = SpotOptim(
            fun=multi_obj,
            bounds=[(-5, 5), (2, 2), (-5, 5)],  # Middle dimension fixed
            max_iter=20,
            n_initial=12,
            seed=42,
            verbose=False
        )
        
        result = optimizer.optimize()
        
        assert result.success
        assert optimizer.y_mo is not None
        assert result.x[1] == 2.0  # Fixed dimension

    def test_mo_with_tensorboard(self):
        """Test multi-objective with TensorBoard logging."""
        def multi_obj(X):
            return np.column_stack([
                np.sum(X**2, axis=1),
                np.sum((X-1)**2, axis=1)
            ])
        
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            optimizer = SpotOptim(
                fun=multi_obj,
                bounds=[(-5, 5), (-5, 5)],
                max_iter=15,
                n_initial=10,
                tensorboard_log=True,
                tensorboard_path=temp_dir,
                seed=42,
                verbose=False
            )
            
            result = optimizer.optimize()
            
            assert result.success
            assert optimizer.y_mo is not None
            # TensorBoard should log the converted single-objective values
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_mo_with_custom_var_names(self):
        """Test multi-objective with custom variable names."""
        def multi_obj(X):
            return np.column_stack([
                np.sum(X**2, axis=1),
                np.sum((X-1)**2, axis=1)
            ])
        
        optimizer = SpotOptim(
            fun=multi_obj,
            bounds=[(-5, 5), (-5, 5)],
            var_name=["alpha", "beta"],
            max_iter=15,
            n_initial=10,
            seed=42,
            verbose=False
        )
        
        result = optimizer.optimize()
        
        assert result.success
        assert optimizer.y_mo is not None
        assert optimizer.var_name == ["alpha", "beta"]
