
"""Tests for the exposed Kriging methods (refactoring check)."""

import numpy as np
import pytest
from spotoptim.surrogate import Kriging

class TestKrigingMethods:
    """Test suite for public Kriging methods."""

    def test_objective_method(self):
        """Test the objective method."""
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        k = Kriging(seed=42).fit(X, y)
        
        # Helper to construct params vector for regression
        # theta (size 1) + Lambda (size 1)
        params = np.concatenate([k.theta_, [k.Lambda_]])
        
        val = k.objective(params)
        
        assert isinstance(val, float)
        assert np.isfinite(val)
        # Should be the same as the stored negLnLike for the optimal params
        assert np.isclose(val, k.negLnLike)

    def test_build_correlation_matrix_method(self):
        """Test the build_correlation_matrix method."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([0.0, 0.25, 1.0])
        k = Kriging(seed=42).fit(X, y)
        
        Psi_upper = k.build_correlation_matrix()
        
        assert isinstance(Psi_upper, np.ndarray)
        assert Psi_upper.shape == (3, 3)
        # It calculates the upper triangle (k=1)
        assert Psi_upper[0, 0] == 0  # Diagonal is handled in likelihood building, this is just upper off-diag
        assert Psi_upper[1, 0] == 0  # Lower triangle should be zero
        assert Psi_upper[0, 1] > 0   # Upper triangle should be populated
        assert Psi_upper[0, 1] <= 1.0 # Correlation <= 1

    def test_likelihood_method(self):
        """Test the likelihood method."""
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        k = Kriging(seed=42).fit(X, y)
        
        params = np.concatenate([k.theta_, [k.Lambda_]])
        nll, Psi, U = k.likelihood(params)
        
        assert isinstance(nll, float)
        assert isinstance(Psi, np.ndarray)
        assert Psi.shape == (2, 2)
        assert U is not None
        assert U.shape == (2, 2)
        
        # Psi should be a valid correlation matrix (symmetric, ones on diagonal + lambda)
        assert np.isclose(Psi[0, 1], Psi[1, 0])
        assert Psi[0, 0] > 1.0  # 1.0 + lambda

    def test_build_psi_vector_method(self):
        """Test the build_psi_vector method."""
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        k = Kriging(seed=42).fit(X, y)
        
        x_new = np.array([0.5]) # Midpoint
        psi = k.build_psi_vector(x_new)
        
        assert isinstance(psi, np.ndarray)
        assert psi.shape == (2,)
        # Midpoint should have equal correlation to both ends (if isotropic/1D)
        assert np.isclose(psi[0], psi[1])
        assert 0 < psi[0] < 1

    def test_predict_single_method(self):
        """Test the predict_single method."""
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        k = Kriging(seed=42).fit(X, y)
        
        x_new = np.array([0.5])
        y_pred, y_std = k.predict_single(x_new)
        
        assert isinstance(y_pred, float)
        assert isinstance(y_std, float)
        assert np.isfinite(y_pred)
        assert y_std >= 0
