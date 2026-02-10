# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from torch.utils.data import TensorDataset, DataLoader
from spotoptim.function.torch_objective import TorchObjective
from spotoptim.core.experiment import ExperimentControl
from spotoptim.core.data import SpotDataFromArray

def test_torch_objective_with_scaler():
    """Test TorchObjective applies scaling correctly."""
    
    # 1. Setup Data
    # Create data with obvious mean/std differences
    # Feature 0: mean=100, std=10
    # Feature 1: mean=0, std=1
    X = torch.normal(mean=100.0, std=10.0, size=(100, 1))
    X2 = torch.normal(mean=0.0, std=1.0, size=(100, 1))
    X_train = torch.cat([X, X2], dim=1).numpy()
    y_train = torch.randn(100, 1).numpy()
    
    dataset = SpotDataFromArray(X_train, y_train)
    
    # 2. Setup Experiment Mock
    # Remove spec to avoid AttributeError if ExperimentControl definition isn't perfectly aligned with our expectation here
    exp = MagicMock() 
    exp.dataset = dataset
    exp.batch_size = 10
    exp.num_workers = 0
    exp.torch_device = "cpu"
    exp.loss_function = None # Use default
    exp.epochs = 1
    
    # Setup hyperparameters mock
    exp.hyperparameters = MagicMock()
    exp.hyperparameters.names.return_value = ["lr"]
    exp.hyperparameters.var_type = ["float"] # Add missing attributes accessed by properties
    exp.hyperparameters.var_name = ["lr"]
    exp.hyperparameters.var_trans = [None]
    exp.hyperparameters.bounds = [(0, 1)]
    exp.metrics = ["val_loss"]
    
    # Mock model class
    model_mock = MagicMock()
    exp.model_class = MagicMock(return_value=model_mock)
    
    # 3. Initialize Objective WITH SCALER
    objective = TorchObjective(exp, use_scaler=True)
    
    assert objective.use_scaler is True
    assert objective.scaler is not None
    
    # 4. Trigger data preparation (where scaling happens)
    train_loader, _ = objective._prepare_data()
    
    # Extract data from loader to verify scaling
    all_x = []
    for x_b, _ in train_loader:
        all_x.append(x_b)
    all_x = torch.cat(all_x)
    
    # Verify scaling
    # Expected mean ~ 0, std ~ 1 for both features
    mean = all_x.mean(dim=0)
    std = all_x.std(dim=0, unbiased=False)
    
    print(f"Original means: {X_train.mean(axis=0)}")
    print(f"Scaled means: {mean}")
    print(f"Scaled stds: {std}")
    
    assert torch.allclose(mean, torch.zeros(2), atol=0.2) # Allow some variance due to randomness
    assert torch.allclose(std, torch.ones(2), atol=0.2)
    
    # Verify that the scaler attributes are set
    assert objective.scaler.mean is not None
    assert objective.scaler.std is not None
    
    # Check approximately correct values in scaler
    assert torch.allclose(objective.scaler.mean, torch.tensor([100.0, 0.0]), atol=2.0)

def test_torch_objective_without_scaler():
    """Verify default behavior remains unchanged."""
    X_train = np.ones((10, 2)) * 100
    y_train = np.ones((10, 1))
    dataset = SpotDataFromArray(X_train, y_train)
    
    exp = MagicMock(spec=ExperimentControl)
    exp.dataset = dataset
    exp.batch_size = 5
    exp.num_workers = 0
    exp.torch_device = "cpu"
    
    objective = TorchObjective(exp, use_scaler=False)
    
    assert objective.use_scaler is False
    assert objective.scaler is None
    
    train_loader, _ = objective._prepare_data()
    
    x_batch, _ = next(iter(train_loader))
    
    # Should maintain original values
    assert torch.allclose(x_batch, torch.tensor(100.0))
