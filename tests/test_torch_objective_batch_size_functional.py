import pytest
import numpy as np
import torch
from unittest.mock import MagicMock
from spotoptim.core.experiment import ExperimentControl
from spotoptim.core.data import SpotDataFromArray
from spotoptim.function.torch_objective import TorchObjective
from spotoptim.hyperparameters import ParameterSet

class MockModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def test_prepare_data_logic():
    """
    Functionally test that _prepare_data creates loaders with the correct batch size
    using real SpotDataFromArray and minimal mocking.
    """
    # 1. Setup Data
    X = np.random.rand(100, 5)
    y = np.random.rand(100, 1)
    dataset = SpotDataFromArray(X, y)
    
    # 2. Setup Experiment
    params = ParameterSet()
    # No params needed here as we test _prepare_data directly
    
    mock_exp = MagicMock(spec=ExperimentControl)
    mock_exp.dataset = dataset
    mock_exp.batch_size = 10 # Default
    mock_exp.num_workers = 0
    mock_exp.hyperparameters = params
    mock_exp.torch_device = "cpu"
    
    objective = TorchObjective(mock_exp)
    
    # 3. Test default
    train_loader, _ = objective._prepare_data()
    assert train_loader.batch_size == 10, "Should use default batch size from experiment"
    
    # 4. Test override
    train_loader_override, _ = objective._prepare_data(batch_size=25)
    assert train_loader_override.batch_size == 25, "Should use provided batch size override"

def test_call_functional_flow():
    """
    Verify the full flow of __call__ passes the correct batch size down to valid loaders.
    """
    X_data = np.random.rand(100, 5)
    y_data = np.random.rand(100, 1)
    dataset = SpotDataFromArray(X_data, y_data)
    
    params = ParameterSet()
    params.add_float("lr", 0.001, 0.1)
    params.add_int("batch_size", 16, 64) 
    
    mock_exp = MagicMock(spec=ExperimentControl)
    mock_exp.dataset = dataset
    mock_exp.batch_size = 32
    mock_exp.num_workers = 0
    mock_exp.hyperparameters = params
    mock_exp.metrics = ["val_loss"]
    mock_exp.epochs = 1
    mock_exp.torch_device = "cpu"
    mock_exp.model_class = MockModel
    mock_exp.loss_function = torch.nn.MSELoss()
    
    objective = TorchObjective(mock_exp)
    
    # We want to intercept train_model to check the loader's batch size
    # instead of running full training (which is slow and tested elsewhere)
    
    def mock_train_model(model, train_loader, val_loader, params):
        # Assertions inside the flow
        expected_bs = int(params["batch_size"])
        assert train_loader.batch_size == expected_bs, f"Expected batch size {expected_bs}, got {train_loader.batch_size}"
        return {"val_loss": 0.1, "epochs": 1.0}
    
    objective.train_model = mock_train_model
    
    # Evaluate with specific batch size
    X_eval = np.array([[0.01, 50.0]])
    objective(X_eval)
