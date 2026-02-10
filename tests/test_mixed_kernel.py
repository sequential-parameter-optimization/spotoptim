# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
from spotoptim.surrogate.kernels import SpotOptimKernel
from spotoptim.surrogate.nystroem import Nystroem

def test_spotoptim_kernel_mixed():
    print("Testing SpotOptimKernel with Mixed Variables...")
    
    # Data: 3 samples, 3 features [float, int, factor]
    X = np.array([
        [1.0, 1, 0],  # Sample 0
        [1.2, 1, 0],  # Sample 1: close to 0
        [5.0, 5, 1]   # Sample 2: far from 0
    ])
    
    var_type = ['float', 'int', 'factor']
    
    # Theta: weights for each feature
    theta = np.array([1.0, 0.5, 2.0]) 
    
    # Instantiate Kernel
    kernel = SpotOptimKernel(theta=theta, var_type=var_type, p_val=2.0, metric_factorial="hamming")
    
    # 1. Test __call__ (correlation matrix)
    K = kernel(X)
    print("Kernel Matrix Shape:", K.shape)
    assert K.shape == (3, 3)
    
    # Diagonal should be 1
    np.testing.assert_allclose(np.diag(K), 1.0)
    print("Diagonal check passed.")
    
    # Sample 0 vs Sample 1 should be high correlation (close distance)
    # Sample 0 vs Sample 2 should be low correlation (far distance)
    print(f"K[0,1] (close): {K[0,1]:.4f}")
    print(f"K[0,2] (far):   {K[0,2]:.4f}")
    assert K[0,1] > K[0,2], "Expected higher correlation for closer samples"
    
def test_nystroem_mixed():
    print("\nTesting Nystroem with SpotOptimKernel (Mixed Variables)...")
    
    # Data: 10 samples, 4 features [float, float, int, factor]
    rng = np.random.RandomState(42)
    n_samples = 20
    X_float = rng.rand(n_samples, 2)
    X_int = rng.randint(0, 5, size=(n_samples, 1))
    X_factor = rng.randint(0, 3, size=(n_samples, 1))
    
    X = np.hstack([X_float, X_int, X_factor])
    var_type = ['float', 'float', 'int', 'factor']
    theta = np.ones(4)
    
    # Define Kernel
    kernel = SpotOptimKernel(theta=theta, var_type=var_type)
    
    # Define Nystroem: 20 samples -> 5 components
    n_components = 5
    nystroem = Nystroem(kernel=kernel, n_components=n_components, random_state=42)
    
    # Fit
    nystroem.fit(X)
    print("Nystroem fitted.")
    print("Components shape:", nystroem.components_.shape)
    assert nystroem.components_.shape == (n_components, 4)
    
    # Transform
    X_trans = nystroem.transform(X)
    print("Transformed Shape:", X_trans.shape)
    assert X_trans.shape == (n_samples, n_components)
    
    # Check if transform output is float (it should be)
    assert X_trans.dtype == np.float64
    print("Nystroem mixed verification successful!")

if __name__ == "__main__":
    test_spotoptim_kernel_mixed()
    test_nystroem_mixed()
