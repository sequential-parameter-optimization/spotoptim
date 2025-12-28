
"""
Tests for tricands integration in SpotOptim.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from spotoptim.SpotOptim import SpotOptim
from scipy.optimize import OptimizeResult

class TestTricandsIntegration:
    """Test suite for tricands acquisition optimizer integration."""

    def setup_method(self):
        """Setup basic optimizer params."""
        self.bounds = [(-5, 5), (0, 10)]
        self.fun = lambda x: np.sum(x**2)
        self.opt = SpotOptim(
            fun=self.fun, 
            bounds=self.bounds,
            acquisition_optimizer="tricands",
            n_initial=5,
            acquisition_fun_return_size=3,
            seed=42
        )
        # Mock surrogate fitting to avoid overhead
        self.opt.model = MagicMock()
        self.opt.model.predict.return_value = (np.zeros(10), np.zeros(10))

    def test_fallback_when_no_points(self):
        """Test fallback to random sampling when not enough points."""
        # Ensure X_ is None or empty
        self.opt.X_ = None
        
        candidates = self.opt.optimize_acquisition_func()
        
        # Should return 'acquisition_fun_return_size' points
        assert candidates.shape == (3, 2)
        # Check bounds
        assert np.all(candidates[:, 0] >= -5)
        assert np.all(candidates[:, 0] <= 5)
        assert np.all(candidates[:, 1] >= 0)
        assert np.all(candidates[:, 1] <= 10)

    def test_tricands_execution(self):
        """Test standard execution with enough points."""
        # Mock X_ with enough points for triangulation (n > m+1)
        self.opt.X_ = np.array([
            [-4, 1], [-2, 8], [0, 5], [2, 2], [4, 9], [1, 1]
        ])
        
        # Mock acquisition function to return predictable values
        # We want to verify that the BEST candidates are chosen.
        # _acquisition_function returns NEGATIVE values.
        # Lets mock it to return values such that specific indices are chosen.
        
        def side_effect(X):
            # Return increasing values. Since we minimize negative acquisition (maximize acq),
            # the lowest values here are "best" (most negative).
            # Wait, usually _acquisition_function returns -1 * EI.
            # So lower is better. 
            # Let's return a range.
            return np.arange(len(X))
            
        self.opt._acquisition_function = MagicMock(side_effect=side_effect)
        
        with patch("spotoptim.SpotOptim.tricands") as mock_tricands:
            # Mock tricands returning 10 candidates in [0, 1] (normalized)
            # We must return normalized candidates because the code expects them from tricands
            mock_cands_norm = np.random.uniform(0, 1, size=(10, 2))
            
            # Set specific values for first 3 to track them.
            # Point 0: [0, 0] -> unnorms to [-5, 0]
            # Point 1: [0.5, 0.5] -> unnorms to [0, 5]
            # Point 2: [1, 1] -> unnorms to [5, 10]
            mock_cands_norm[0] = [0.0, 0.0]
            mock_cands_norm[1] = [0.5, 0.5]
            mock_cands_norm[2] = [1.0, 1.0]
            
            mock_tricands.return_value = mock_cands_norm
            
            candidates = self.opt.optimize_acquisition_func()
            
            # Verify tricands called with normalized data
            mock_tricands.assert_called_once()
            args, kwargs = mock_tricands.call_args
            # Check input X was normalized
            # Start of X_ is [-4, 1]. In bounds [-5, 5], [0, 10]:
            # x1 norm: (-4 - -5)/10 = 0.1
            # x2 norm: (1 - 0)/10 = 0.1
            np.testing.assert_allclose(args[0][0], [0.1, 0.1])
            assert kwargs['nmax'] >= 100 * 2
            
            # Verify denormalization of result
            # We mocked _acquisition_function to return 0, 1, 2... for the candidates.
            # So indices 0, 1, 2 should be selected.
            expected_0 = np.array([-5.0, 0.0])
            expected_1 = np.array([0.0, 5.0])
            expected_2 = np.array([5.0, 10.0])
            
            assert len(candidates) == 3
            np.testing.assert_allclose(candidates[0], expected_0)
            np.testing.assert_allclose(candidates[1], expected_1)
            np.testing.assert_allclose(candidates[2], expected_2)

    def test_nmax_scaling(self):
        """Test that nmax scales with acquisition_fun_return_size."""
        self.opt.acquisition_fun_return_size = 50
        self.opt.X_ = np.random.uniform(0, 1, size=(10, 2))
        
        with patch("spotoptim.SpotOptim.tricands") as mock_tricands:
            mock_tricands.return_value = np.zeros((100, 2))
            self.opt._acquisition_function = MagicMock(return_value=np.zeros(100))
            
            self.opt.optimize_acquisition_func()
            
            # nmax should be max(100*2, 50*50) = 2500
            _, kwargs = mock_tricands.call_args
            assert kwargs['nmax'] >= 2500
