import numpy as np
import pytest
import torch
from spotoptim.core.data import SpotDataFromArray
from spotoptim.core.experiment import ExperimentControl
from spotoptim.hyperparameters.parameters import ParameterSet
from spotoptim.function.torch_objective import TorchObjective
from spotoptim.nn.linear_regressor import LinearRegressor

def test_torch_objective_linear_regressor():
    # Setup data
    X = np.random.randn(20, 5).astype(np.float32)
    y = np.random.randn(20, 1).astype(np.float32)
    data = SpotDataFromArray(X, y) # Default creates train only, no val
    
    # Setup hyperparameters
    params = ParameterSet()
    params.add_float("lr", 1e-3, 1e-1, transform="log")
    params.add_int("l1", 16, 32)
    params.add_int("num_hidden_layers", 1, 2)
    params.add_categorical("activation", ["ReLU", "Tanh"])
    params.add_categorical("optimizer", ["Adam", "SGD"])

    # Setup experiment
    exp = ExperimentControl(
        dataset=data,
        model_class=LinearRegressor,
        hyperparameters=params,
        epochs=2, # short for testing
        batch_size=4,
        metrics=["mse"]
    )
    
    objective = TorchObjective(exp)
    
    # Create a sample configuration vector
    # lr=0.01, l1=32, layers=1, activation='ReLU' (idx 0), optimizer='Adam' (idx 0)
    # Note: Categorical bounds in SpotOptim are handled, but here valid values are expected.
    # SpotOptim would pass indices for factors usually if var_type is factor?
    # Wait, my TorchObjective decodes using:
    # idx = int(val) -> choices[idx]
    # So we should pass indices for categorical variables.
    
    # Bounds logic in SpotOptim: (0, n_levels-1) for factor.
    # So valid X values are:
    # float, int, int (index), int (index)
    
    # lr=0.01, l1=16, layers=1, activation=0, optimizer=0
    x_input = np.array([0.01, 16.0, 1.0, 0.0, 0.0])
    
    # Test __call__
    res = objective(x_input)
    
    assert res.shape == (1, 1)
    assert np.isfinite(res).all()
    # Loss should be positive
    assert res[0, 0] >= 0

def test_torch_objective_with_validation():
    # Setup data with validation
    X = np.random.randn(20, 5).astype(np.float32)
    y = np.random.randn(20, 1).astype(np.float32)
    X_val = np.random.randn(10, 5).astype(np.float32)
    y_val = np.random.randn(10, 1).astype(np.float32)
    
    data = SpotDataFromArray(X, y, x_val=X_val, y_val=y_val)
    
    params = ParameterSet().add_float("lr", 1e-3, 1e-1)
    
    exp = ExperimentControl(
        dataset=data,
        model_class=LinearRegressor,
        hyperparameters=params,
        epochs=2
    )
    
    objective = TorchObjective(exp)
    res = objective(np.array([0.01]))
    
    assert res.shape == (1, 1)
    assert np.isfinite(res).all()
