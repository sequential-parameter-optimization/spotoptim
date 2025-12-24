import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from spotoptim.core.experiment import ExperimentControl
from spotoptim.function.torch_objective import TorchObjective
from spotoptim.hyperparameters import ParameterSet

class MockModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def test_batch_size_tuning():
    """
    Test that batch_size is correctly extracted from hyperparameters and used
    specifying the DataLoader batch size during evaluation.
    """
    # 1. Setup Mock Experiment
    # Define parameters INCLUDING batch_size
    params = ParameterSet()
    params.add_float("lr", 0.001, 0.1)
    params.add_int("batch_size", 16, 128) # Tunable batch size

    mock_exp = MagicMock(spec=ExperimentControl)
    mock_exp.hyperparameters = params
    mock_exp.metrics = ["val_loss"]
    mock_exp.epochs = 1
    mock_exp.torch_device = "cpu"
    mock_exp.batch_size = 32 # Default batch size
    mock_exp.num_workers = 0
    
    # Mock Dataset
    mock_dataset = MagicMock()
    mock_dataset.input_dim = 10
    mock_dataset.output_dim = 1
    mock_exp.dataset = mock_dataset
    mock_exp.model_class = MockModel

    # 2. Initialize Objective
    objective = TorchObjective(mock_exp)

    # 3. Mock _prepare_data to verify calls without actual data loading overhead
    # We return dummy loaders
    train_loader = MagicMock()
    val_loader = MagicMock()
    
    with patch.object(objective, '_prepare_data', return_value=(train_loader, val_loader)) as mock_prepare:
        # Mock train_model to avoid actual training execution
        with patch.object(objective, 'train_model', return_value={"val_loss": 0.5, "epochs": 1}):
            
            # 4. Evaluate with a specific batch size
            # X corresponds to [lr, batch_size]
            # Let's say lr=0.01, batch_size=64
            X_eval = np.array([[0.01, 64.0]])
            
            # Call objective
            objective(X_eval)
            
            # 5. Verify _prepare_data was called with batch_size=64
            # It might be called once initially with no args (default) and then with specific args
            
            # Check calls
            # The first call is self._prepare_data() with no args (setup)
            # The second call should be self._prepare_data(batch_size=64) inside the loop
            
            # Filter calls with keyword args
            kw_calls = [call for call in mock_prepare.mock_calls if 'batch_size' in call.kwargs]
            
            assert len(kw_calls) > 0, "_prepare_data should have been called with batch_size kwarg"
            
            last_call_kwargs = kw_calls[-1].kwargs
            assert last_call_kwargs['batch_size'] == 64, f"Expected batch_size=64, got {last_call_kwargs['batch_size']}"

def test_fixed_batch_size():
    """
    Test that if batch_size is NOT in hyperparameters, it uses the experiment's fixed batch size
    and does NOT call _prepare_data repeatedly with an argument.
    """
    # 1. Setup Mock Experiment WITHOUT batch_size in params
    params = ParameterSet()
    params.add_float("lr", 0.001, 0.1)
    # batch_size is NOT added

    mock_exp = MagicMock(spec=ExperimentControl)
    mock_exp.hyperparameters = params
    mock_exp.metrics = ["val_loss"]
    mock_exp.epochs = 1
    mock_exp.torch_device = "cpu"
    mock_exp.batch_size = 32 
    mock_exp.num_workers = 0
    
    mock_dataset = MagicMock()
    mock_dataset.input_dim = 10
    mock_dataset.output_dim = 1
    mock_exp.dataset = mock_dataset
    mock_exp.model_class = MockModel

    objective = TorchObjective(mock_exp)

    # Mock _prepare_data
    train_loader = MagicMock()
    val_loader = MagicMock()
    
    with patch.object(objective, '_prepare_data', return_value=(train_loader, val_loader)) as mock_prepare:
        with patch.object(objective, 'train_model', return_value={"val_loss": 0.5, "epochs": 1}):
            
            X_eval = np.array([[0.01]])
            objective(X_eval)
            
            # _prepare_data should be called exactly once (initial setup)
            assert mock_prepare.call_count == 1
            # And it should be called WITHOUT batch_size argument (or None implicitly)
            args, kwargs = mock_prepare.call_args
            assert 'batch_size' not in kwargs or kwargs['batch_size'] is None
