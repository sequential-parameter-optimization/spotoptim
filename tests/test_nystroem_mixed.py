# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pytest
from spotoptim.surrogate.kernels import SpotOptimKernel
from spotoptim.surrogate.nystroem import Nystroem
from spotoptim.surrogate.kriging import Kriging
from spotoptim.surrogate.pipeline import Pipeline

def test_spotoptim_kernel_shapes():
    """Test that SpotOptimKernel returns correct shapes and diagonal."""
    # Data: 3 samples, 3 features [float, int, factor]
    X = np.array([
        [1.0, 1, 0],
        [1.2, 1, 0],
        [5.0, 5, 1]
    ])
    var_type = ['float', 'int', 'factor']
    theta = np.array([1.0, 0.5, 2.0])
    
    kernel = SpotOptimKernel(theta=theta, var_type=var_type, p_val=2.0)
    K = kernel(X)
    
    assert K.shape == (3, 3)
    np.testing.assert_allclose(np.diag(K), 1.0)

def test_spotoptim_kernel_values():
    """Test specific correlation logic for mixed variables."""
    # 2 samples, 2 features [float, factor]
    # theta = [1, 1]
    X = np.array([
        [0.0, 0],
        [1.0, 1]
    ])
    var_type = ['float', 'factor']
    theta = np.array([1.0, 1.0])
    
    kernel = SpotOptimKernel(theta=theta, var_type=var_type)
    K = kernel(X)
    
    # Distance:
    # float: (0-1)^2 * 1.0 = 1.0
    # factor: (0!=1) -> 1 * 1.0 = 1.0
    # Total D = 2.0
    # Correlation = exp(-2.0)
    expected_corr = np.exp(-2.0)
    
    np.testing.assert_allclose(K[0, 1], expected_corr)

def test_nystroem_mixed_integration():
    """Test Nystroem works with SpotOptimKernel on mixed data."""
    rng = np.random.RandomState(42)
    n_samples = 20
    # 2 floats, 1 int, 1 factor
    X = np.hstack([
        rng.rand(n_samples, 2),
        rng.randint(0, 5, size=(n_samples, 1)),
        rng.randint(0, 3, size=(n_samples, 1))
    ])
    var_type = ['float', 'float', 'int', 'factor']
    theta = np.ones(4)
    
    kernel = SpotOptimKernel(theta=theta, var_type=var_type)
    n_components = 5
    nystroem = Nystroem(kernel=kernel, n_components=n_components, random_state=42)
    
    nystroem.fit(X)
    assert nystroem.components_.shape == (n_components, 4)
    
    X_trans = nystroem.transform(X)
    assert X_trans.shape == (n_samples, n_components)
    assert X_trans.dtype == np.float64

def test_nystroem_pipeline_kriging_mixed():
    """
    Test end-to-end pipeline with Nystroem (Mixed) -> Kriging.
    
    Note: The Kriging model receives continuous features from Nystroem,
    so we configure Kriging as if it's receiving 'float' variables.
    """
    rng = np.random.RandomState(42)
    n_samples = 30
    n_features = 4
    
    # Mixed input data
    X = np.hstack([
        rng.rand(n_samples, 2),
        rng.randint(0, 5, size=(n_samples, 1)),
        rng.randint(0, 3, size=(n_samples, 1))
    ])
    y = np.sum(X[:, :2], axis=1) # Simple target
    
    var_type_input = ['float', 'float', 'int', 'factor']
    theta_input = np.ones(4)
    
    # 1. Nystroem with Mixed Kernel
    kernel = SpotOptimKernel(theta=theta_input, var_type=var_type_input)
    nystroem = Nystroem(kernel=kernel, n_components=10, random_state=42)
    
    # 2. Kriging on transformed features
    # Transformed features are 10 floats
    gp = Kriging(seed=42)
    
    pipeline = Pipeline([
        ('nystroem', nystroem),
        ('gp', gp)
    ])
    
    pipeline.fit(X, y)
    
    # Predict on new data
    X_test = np.hstack([
        rng.rand(5, 2),
        rng.randint(0, 5, size=(5, 1)),
        rng.randint(0, 3, size=(5, 1))
    ])
    y_pred = pipeline.predict(X_test)
    
    assert y_pred.shape == (5,)
    assert not np.isnan(y_pred).any()
