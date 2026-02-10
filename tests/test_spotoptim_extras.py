# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import unittest
import numpy as np
from spotoptim import SpotOptim

class TestSpotOptimExtras(unittest.TestCase):
    def test_get_best_hyperparameters_basic(self):
        def objective(X):
            # Simple sphere function
            return np.sum(X**2, axis=1)
            
        bounds = [(-1, 1), (-1, 1)]
        optimizer = SpotOptim(fun=objective, bounds=bounds, max_iter=10, n_initial=5, var_name=["x", "y"], seed=42)
        optimizer.optimize()
        
        # Test get_best_hyperparameters
        best_params = optimizer.get_best_hyperparameters()
        self.assertIsInstance(best_params, dict)
        self.assertIn("x", best_params)
        self.assertIn("y", best_params)
        
        # Values should match best_x_
        self.assertAlmostEqual(best_params["x"], optimizer.best_x_[0])
        self.assertAlmostEqual(best_params["y"], optimizer.best_x_[1])

    def test_get_best_hyperparameters_with_noise(self):
        def noisy_objective(X):
             # Simple sphere function with noise
            return np.sum(X**2, axis=1) + np.random.normal(0, 0.1, size=X.shape[0])

        bounds = [(-1, 1)]
        # Use repeats to trigger noise handling
        optimizer = SpotOptim(
            fun=noisy_objective, 
            bounds=bounds, 
            max_iter=10, 
            n_initial=5, 
            var_name=["x"], 
            repeats_initial=2,
            seed=42
        )
        optimizer.optimize()
        
        best_params = optimizer.get_best_hyperparameters()
        # Should use min_mean_X because of noise
        self.assertAlmostEqual(best_params["x"], optimizer.min_mean_X[0])

if __name__ == "__main__":
    unittest.main()
