
"""
Tests for flexible acquisition optimizer in SpotOptim.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from scipy.optimize import OptimizeResult
from spotoptim.SpotOptim import SpotOptim

class TestAcquisitionOptimizer:
    """Test suite for acquisition optimizer flexibility."""

    def setup_method(self):
        """Setup basic optimizer params."""
        self.bounds = [(-5, 5), (-5, 5)]
        self.fun = lambda x: np.sum(x**2)
        
    def test_default_optimizer_is_differential_evolution(self):
        """Test that default optimizer is DE."""
        opt = SpotOptim(fun=self.fun, bounds=self.bounds)
        assert opt.acquisition_optimizer == "differential_evolution"
        
        # Verify DE is called
        with patch("spotoptim.SpotOptim.differential_evolution") as mock_de:
            mock_de.return_value = OptimizeResult(x=np.array([1.0, 1.0]))
            # Handle acquisition function mocking internal call
            opt._acquisition_function = MagicMock(return_value=0.0)
            
            opt.optimize_acquisition_func()
            mock_de.assert_called_once()
            
    def test_minimize_optimizer_nelder_mead(self):
        """Test using Nelder-Mead via minimize interface."""
        opt = SpotOptim(
            fun=self.fun, 
            bounds=self.bounds, 
            acquisition_optimizer="Nelder-Mead",
            seed=42,
            acquisition_fun_return_size=1
        )
        
        with patch("spotoptim.SpotOptim.minimize") as mock_minimize:
            mock_minimize.return_value = OptimizeResult(x=np.array([0.5, 0.5]))
            opt._acquisition_function = MagicMock(return_value=0.0)
            
            res = opt.optimize_acquisition_func()
            
            mock_minimize.assert_called_once()
            args, kwargs = mock_minimize.call_args
            assert kwargs["method"] == "Nelder-Mead"
            assert "x0" in kwargs
            # Verify x0 is within bounds
            assert np.all(kwargs["x0"] >= -5)
            assert np.all(kwargs["x0"] <= 5)
            
            np.testing.assert_array_equal(res, np.array([0.5, 0.5]))

    def test_custom_callable_optimizer(self):
        """Test using a custom callable optimizer."""
        
        def custom_optimizer(fun, x0, bounds, **kwargs):
            # Mock optimizer that always returns [2.0, 2.0]
            return OptimizeResult(x=np.array([2.0, 2.0]))
            
        opt = SpotOptim(
            fun=self.fun, 
            bounds=self.bounds, 
            acquisition_optimizer=custom_optimizer,
            acquisition_fun_return_size=1
        )
        
        opt._acquisition_function = MagicMock(return_value=0.0)
        
        res = opt.optimize_acquisition_func()
        np.testing.assert_array_equal(res, np.array([2.0, 2.0]))

    def test_invalid_optimizer_type(self):
        """Test exception for invalid optimizer type."""
        opt = SpotOptim(
            fun=self.fun, 
            bounds=self.bounds, 
            acquisition_optimizer=123 # Invalid int
        )
        
        opt._acquisition_function = MagicMock(return_value=0.0)
        
        with pytest.raises(ValueError, match="Unknown acquisition optimizer type"):
            opt.optimize_acquisition_func()

    def test_minimize_optimizer_with_return_size_gt_1(self):
        """Test that minimize optimizer returns 2D array when return_size > 1."""
        opt = SpotOptim(
            fun=self.fun, 
            bounds=self.bounds, 
            acquisition_optimizer="Nelder-Mead",
            acquisition_fun_return_size=3
        )
        
        with patch("spotoptim.SpotOptim.minimize") as mock_minimize:
            mock_minimize.return_value = OptimizeResult(x=np.array([0.5, 0.5]))
            opt._acquisition_function = MagicMock(return_value=0.0)
            
            res = opt.optimize_acquisition_func()
            
            # Should be shape (1, 2) event though size=3 asked,
            # because minimize only returns 1 result.
            assert res.shape == (1, 2)
            np.testing.assert_array_equal(res, np.array([[0.5, 0.5]]))
