"""
Tests for acquisition_fun_return_size functionality in SpotOptim.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from spotoptim.SpotOptim import SpotOptim
from scipy.optimize import OptimizeResult

class TestAcquisitionReturnSize:
    """Test suite for acquisition_fun_return_size parameter."""

    def test_init_sets_parameter(self):
        """Test that __init__ correctly sets acquisition_fun_return_size."""
        opt = SpotOptim(fun=lambda x: np.sum(x**2), bounds=[(-1, 1)], acquisition_fun_return_size=3)
        assert opt.acquisition_fun_return_size == 3
        
        opt_default = SpotOptim(fun=lambda x: np.sum(x**2), bounds=[(-1, 1)])
        assert opt_default.acquisition_fun_return_size == 3

    @patch("spotoptim.SpotOptim.differential_evolution")
    def test_optimize_acquisition_func_returns_single(self, mock_de):
        """Test default behavior returns 1D array."""
        mock_res = MagicMock(spec=OptimizeResult)
        mock_res.x = np.array([1.0, 2.0])
        mock_de.return_value = mock_res
        
        opt = SpotOptim(fun=lambda x: np.sum(x**2), bounds=[(-1, 1), (-1, 1)], acquisition_fun_return_size=1)
        # Mock acquisition function to avoid call logic issues
        opt._acquisition_function = MagicMock(return_value=0.0)
        
        res = opt.optimize_acquisition_func()
        assert res.shape == (2,)
        np.testing.assert_array_equal(res, mock_res.x)

    @patch("spotoptim.SpotOptim.differential_evolution")
    def test_optimize_acquisition_func_returns_multiple(self, mock_de):
        """Test returns top N candidates when size > 1."""
        # Setup mock behavior to simulate callback callback
        population = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        # Energies: 1st is best (lowest), 3rd is second best, 2nd is third...
        # Let's make it clear: [10, 30, 20, 40] -> sort args: [0, 2, 1, 3] -> best indices: 0, 2, 1
        energies = np.array([10.0, 30.0, 20.0, 40.0])
        
        def side_effect(*args, **kwargs):
            callback = kwargs.get("callback")
            if callback:
                intermediate = MagicMock(spec=OptimizeResult)
                intermediate.population = population
                intermediate.population_energies = energies
                callback(intermediate)
            
            res = MagicMock(spec=OptimizeResult)
            res.x = population[0] # Best
            return res
            
        mock_de.side_effect = side_effect
        
        opt = SpotOptim(
            fun=lambda x: np.sum(x**2), 
            bounds=[(-5, 5), (-5, 5)], 
            acquisition_fun_return_size=3
        )
        opt._acquisition_function = MagicMock(return_value=0.0)
        
        candidates = opt.optimize_acquisition_func()
        
        # Expecting top 3 sorted by energy: indices 0 (10.0), 2 (20.0), 1 (30.0)
        expected_indices = [0, 2, 1]
        expected_candidates = population[expected_indices]
        
        assert candidates.shape == (3, 2)
        np.testing.assert_array_equal(candidates, expected_candidates)

    @patch("spotoptim.SpotOptim.differential_evolution")
    def testsuggest_next_infill_point_uses_alternatives(self, mock_de):
        """Test suggest_next_infill_point tries 2nd candidate if 1st is duplicate."""
        # Setup: 1st candidate is duplicate, 2nd is new
        existing_point = np.array([1.0, 1.0])
        new_point = np.array([2.0, 2.0])
        
        population = np.array([existing_point, new_point]) # energies implied sorted
        energies = np.array([0.1, 0.2])
        
        def side_effect(*args, **kwargs):
            callback = kwargs.get("callback")
            if callback:
                intermediate = MagicMock()
                intermediate.population = population
                intermediate.population_energies = energies
                callback(intermediate)
            return MagicMock(x=existing_point)
            
        mock_de.side_effect = side_effect
        
        opt = SpotOptim(
            fun=lambda x: np.sum(x**2), 
            bounds=[(-5, 5), (-5, 5)], 
            acquisition_fun_return_size=2,
            var_type=['float', 'float']
        )
        opt.X_ = existing_point.reshape(1, 2) # 1st is in history
        opt._acquisition_function = MagicMock(return_value=0.0)
        opt.tolerance_x = 0.1
        
        # Should skip 1st (duplicate) and return 2nd
        x_suggested = opt.suggest_next_infill_point()
        
        np.testing.assert_array_almost_equal(x_suggested, new_point)
