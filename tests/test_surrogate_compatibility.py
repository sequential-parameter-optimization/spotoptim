"""Tests for surrogate model compatibility with different regressors."""

import numpy as np
import pytest
from spotoptim import SpotOptim
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR


def sphere(X):
    """Simple sphere function for testing."""
    return np.sum(X**2, axis=1)


@pytest.mark.parametrize("surrogate_model,name", [
    (RandomForestRegressor(n_estimators=10, random_state=42), "RandomForest"),
    (GradientBoostingRegressor(n_estimators=10, random_state=42), "GradientBoosting"),
    (SVR(kernel='rbf'), "SVR"),
])
def test_surrogate_without_uncertainty(surrogate_model, name):
    """Test that surrogates without return_std work correctly."""
    bounds = [(-5, 5), (-5, 5)]
    
    opt = SpotOptim(
        fun=sphere,
        bounds=bounds,
        surrogate=surrogate_model,
        max_iter=15,
        n_initial=5,
        acquisition='y',  # Use greedy for non-GP surrogates
        seed=42,
        verbose=False
    )
    
    result = opt.optimize()
    
    # Check that optimization completed
    assert result.success
    assert result.nfev == 15
    assert result.fun < 10.0  # Should find something reasonable
    
    # Check that plotting methods work without errors
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    opt.plot_progress(show=False)
    opt.plot_important_hyperparameter_contour(max_imp=2, show=False)
    opt.plot_surrogate(i=0, j=1, show=False)


def test_predict_with_uncertainty_helper():
    """Test the _predict_with_uncertainty helper method."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
    
    bounds = [(-5, 5), (-5, 5)]
    
    # Test with GP (has return_std)
    gp = GaussianProcessRegressor(kernel=RBF(), random_state=42)
    opt_gp = SpotOptim(
        fun=sphere,
        bounds=bounds,
        surrogate=gp,
        max_iter=10,
        n_initial=5,
        seed=42,
        verbose=False
    )
    opt_gp.optimize()
    
    X_test = np.array([[0.0, 0.0], [1.0, 1.0]])
    pred, std = opt_gp._predict_with_uncertainty(X_test)
    
    assert pred.shape == (2,)
    assert std.shape == (2,)
    assert np.all(std >= 0)  # Std should be non-negative
    
    # Test with RF (no return_std)
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    opt_rf = SpotOptim(
        fun=sphere,
        bounds=bounds,
        surrogate=rf,
        max_iter=10,
        n_initial=5,
        acquisition='y',
        seed=42,
        verbose=False
    )
    opt_rf.optimize()
    
    pred_rf, std_rf = opt_rf._predict_with_uncertainty(X_test)
    
    assert pred_rf.shape == (2,)
    assert std_rf.shape == (2,)
    assert np.all(std_rf == 0)  # Should return zeros for std


def test_acquisition_with_non_gp_surrogate():
    """Test that EI/PI acquisitions gracefully handle non-GP surrogates."""
    bounds = [(-5, 5), (-5, 5)]
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    
    # Using 'ei' with RF should work (will get zero std)
    opt = SpotOptim(
        fun=sphere,
        bounds=bounds,
        surrogate=rf,
        max_iter=10,
        n_initial=5,
        acquisition='ei',  # This should work but behave like greedy
        seed=42,
        verbose=False
    )
    
    result = opt.optimize()
    assert result.success
    
    # PI should also work
    opt_pi = SpotOptim(
        fun=sphere,
        bounds=bounds,
        surrogate=rf,
        max_iter=10,
        n_initial=5,
        acquisition='pi',
        seed=42,
        verbose=False
    )
    
    result_pi = opt_pi.optimize()
    assert result_pi.success
