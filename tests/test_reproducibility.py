
import unittest
import torch
import torch.nn as nn
import numpy as np
from spotoptim.core.experiment import ExperimentControl
from spotoptim.function.torch_objective import TorchObjective
from spotoptim.hyperparameters import ParameterSet
from spotoptim.core.data import SpotDataFromArray

class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.fc(x)

class TestReproducibility(unittest.TestCase):
    def test_torch_objective_seeding(self):
        # 1. Prepare data
        X_data = np.random.rand(10, 2)
        y_data = np.random.rand(10, 1)
        dataset = SpotDataFromArray(X_data, y_data)

        # 2. Define hyperparameters
        params = ParameterSet()
        params.add_float("lr", 1e-4, 1e-2, default=1e-3)

        # 3. Setup Experiment
        exp = ExperimentControl(
            experiment_name="test_repro",
            model_class=SimpleModel,
            dataset=dataset,
            hyperparameters=params,
            metrics=["val_loss"],
            epochs=1,
            batch_size=2,
            seed=42  # Experiment has a seed
        )

        # 4. Initialize Objective
        # Should pick up seed from experiment by default
        objective = TorchObjective(exp)
        
        X_eval = np.array([[0.005]])
        
        # Run 1
        y_eval1 = objective(X_eval)
        
        # Run 2
        y_eval2 = objective(X_eval)
        
        np.testing.assert_allclose(y_eval1, y_eval2, err_msg="Results should be identical with seeding")

    def test_torch_objective_explicit_seed(self):
        # Test passing seed explicitly to TorchObjective
        X_data = np.random.rand(10, 2)
        y_data = np.random.rand(10, 1)
        dataset = SpotDataFromArray(X_data, y_data)
        params = ParameterSet().add_float("lr", 1e-4, 1e-2)
        
        exp = ExperimentControl(
            experiment_name="test_explicit_seed",
            model_class=SimpleModel,
            dataset=dataset,
            hyperparameters=params,
            seed=None # No seed in experiment
        )
        
        # Pass seed here
        objective = TorchObjective(exp, seed=999)
        
        X_eval = np.array([[0.005]])
        y_eval1 = objective(X_eval)
        y_eval2 = objective(X_eval)
        
        np.testing.assert_allclose(y_eval1, y_eval2, err_msg="Results should be identical with explicit seeding")

if __name__ == "__main__":
    unittest.main()
