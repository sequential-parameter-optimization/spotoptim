import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from spotoptim.function.torch_objective import TorchObjective
from spotoptim.core.experiment import ExperimentControl
from spotoptim.hyperparameters import ParameterSet
from spotoptim.core.data import SpotDataFromArray
from torch.utils.data import DataLoader

# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units=10, **kwargs):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.kwargs = kwargs

    def forward(self, x):
        return self.fc(x)

@pytest.fixture
def mock_experiment():
    """Create a mock ExperimentControl with necessary attributes."""
    exp = MagicMock(spec=ExperimentControl)
    
    # Setup Hyperparameters
    # Using real ParameterSet might be easier, but mocking ensures isolation
    # Let's mock the internal structure of ParameterSet used by TorchObjective
    param_set = MagicMock(spec=ParameterSet)
    param_set.bounds = [(-5.0, 5.0), (1, 10)]
    param_set.var_type = ["float", "int"]
    param_set.var_name = ["x1", "epochs"]
    param_set.var_trans = ["linear", "linear"]
    param_set.names.return_value = ["x1", "epochs"]
    param_set._var_types = ["float", "int"]
    
    # Mocking _parameters list for factor handling (though not used in this basic setup)
    param_set._parameters = [{}, {}]

    exp.hyperparameters = param_set
    exp.metrics = ["val_loss", "epochs"]
    exp.torch_device = torch.device("cpu")
    exp.loss_function = nn.MSELoss()
    exp.epochs = 5
    exp.batch_size = 2
    exp.num_workers = 0
    exp.model_class = SimpleModel
    
    # Setup Dataset
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 1)
    dataset = MagicMock(spec=SpotDataFromArray)
    dataset.get_train_data.return_value = (X, y)
    dataset.get_validation_data.return_value = (X, y) # Same for simplicity
    dataset.input_dim = 2
    dataset.output_dim = 1
    
    exp.dataset = dataset
    
    return exp

def test_init_and_properties(mock_experiment):
    """Test initialization and property access."""
    obj = TorchObjective(mock_experiment)
    
    assert obj.experiment == mock_experiment
    assert obj.device == mock_experiment.torch_device
    assert obj.bounds == [(-5.0, 5.0), (1, 10)]
    assert obj.var_type == ["float", "int"]
    assert obj.var_name == ["x1", "epochs"]
    assert obj.objective_names == ["val_loss", "epochs"]

def test_get_hyperparameters(mock_experiment):
    """Test hyperparameter decoding."""
    obj = TorchObjective(mock_experiment)
    
    # Test float and int casting
    X = np.array([0.5, 5.7]) # 5.7 should round to 6 for int type
    params = obj._get_hyperparameters(X)
    
    assert params["x1"] == 0.5
    assert params["epochs"] == 6
    assert isinstance(params["epochs"], int)

def test_prepare_data(mock_experiment):
    """Test DataLoader creation."""
    obj = TorchObjective(mock_experiment)
    train_loader, val_loader = obj._prepare_data()
    
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert train_loader.batch_size == 2

def test_train_model(mock_experiment):
    """Test the training loop."""
    obj = TorchObjective(mock_experiment)
    train_loader, val_loader = obj._prepare_data()
    
    model = SimpleModel(2, 1)
    params = {"lr": 0.01, "epochs": 2}
    
    metrics = obj.train_model(model, train_loader, val_loader, params)
    
    assert "val_loss" in metrics
    assert "train_loss" in metrics
    assert "mse" in metrics
    assert metrics["epochs"] == 2.0

def test_call_integration(mock_experiment):
    """Test the full __call__ execution."""
    obj = TorchObjective(mock_experiment)
    
    # Input with 2 samples
    X = np.array([
        [0.1, 5],
        [0.2, 5]
    ])
    
    # Mock train_model to avoid actual training time and ensure deterministic return
    with patch.object(TorchObjective, 'train_model') as mock_train:
        mock_train.side_effect = [
            {"val_loss": 0.1, "epochs": 5.0},
            {"val_loss": 0.2, "epochs": 5.0}
        ]
        
        results = obj(X)
        
        # Check calls
        assert mock_train.call_count == 2
        
        # Check results shape and content
        # Metrics are ["val_loss", "epochs"]
        assert results.shape == (2, 2)
        assert results[0, 0] == 0.1 # val_loss
        assert results[0, 1] == 5.0 # epochs
        assert results[1, 0] == 0.2
