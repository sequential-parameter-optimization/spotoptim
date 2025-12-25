
import torch
import torch.nn as nn
import numpy as np
from spotoptim.core.experiment import ExperimentControl
from spotoptim.function.torch_objective import TorchObjective
from spotoptim.hyperparameters import ParameterSet
from spotoptim.core.data import SpotDataFromArray
import sys

def run_repro_test():
    # 1. Define a simple model (MLP-like but simple)
    class SimpleModel(nn.Module):
        def __init__(self, input_dim, output_dim, **kwargs):
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim)
        def forward(self, x):
            return self.fc(x)

    # 2. Prepare data
    X_data = np.random.rand(10, 2)
    y_data = np.random.rand(10, 1)
    dataset = SpotDataFromArray(X_data, y_data)

    # 3. Define hyperparameters
    params = ParameterSet()
    params.add_float("lr", 1e-4, 1e-2, default=1e-3)

    # 4. Setup Experiment
    exp = ExperimentControl(
        experiment_name="test_exp",
        model_class=SimpleModel,
        dataset=dataset,
        hyperparameters=params,
        metrics=["val_loss"],
        epochs=1,
        batch_size=2
    )

    # 5. Initialize/Instantiate Objective
    objective = TorchObjective(exp)
    
    # 6. Evaluate twice with same input
    X_eval = np.array([[0.005]])
    
    # Run 1
    print("Run 1:")
    y_eval1 = objective(X_eval)
    print(f"y_eval1: {y_eval1}")
    
    # Run 2 (without re-seeding explicitly in user code, mimicking user complaint)
    print("Run 2:")
    y_eval2 = objective(X_eval)
    print(f"y_eval2: {y_eval2}")
    
    if np.allclose(y_eval1, y_eval2):
        print("PASS: Results are identical (Reproducible).")
    else:
        print("FAIL: Results are different (Non-reproducible).")

if __name__ == "__main__":
    run_repro_test()
