# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Schedule-free optimizer integration in TorchObjective and MLP.get_optimizer."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock

from spotoptim.core.data import SpotDataFromArray
from spotoptim.core.experiment import ExperimentControl
from spotoptim.function.torch_objective import TorchObjective
from spotoptim.hyperparameters import ParameterSet
from spotoptim.nn.mlp import MLP
from spotoptim.optimizer import AdamWScheduleFree
from spotoptim.utils.mapping import map_lr


class RecordingAdamWScheduleFree(AdamWScheduleFree):
    """AdamWScheduleFree that counts train()/eval() calls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_calls = 0
        self.eval_calls = 0

    def train(self):
        self.train_calls += 1
        super().train()

    def eval(self):
        self.eval_calls += 1
        super().eval()


class SpyModel(nn.Module):
    """Minimal model whose get_optimizer returns a recording schedule-free optimizer."""

    def __init__(self, input_dim=2, output_dim=1, **kwargs):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.optimizer = None

    def forward(self, x):
        return self.fc(x)

    def get_optimizer(self, optimizer_name="Adam", lr=None, **kwargs):
        self.optimizer = RecordingAdamWScheduleFree(self.parameters(), lr=0.0025)
        return self.optimizer


def _mock_experiment(epochs=3):
    exp = MagicMock()
    exp.loss_function = nn.MSELoss()
    exp.epochs = epochs
    exp.torch_device = torch.device("cpu")
    return exp


def _loaders(n=12, features=2, batch_size=4):
    torch.manual_seed(0)
    X = torch.randn(n, features)
    y = torch.randn(n, 1)
    train_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size)
    val_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size)
    return train_loader, val_loader


def test_map_lr_has_adamw_schedule_free_entry():
    assert map_lr(1.0, "AdamWScheduleFree") == 0.0025
    assert map_lr(2.0, "AdamWScheduleFree") == 0.005


def test_mlp_get_optimizer_maps_unified_lr():
    """lr=1.0 must map to the schedule-free default 0.0025, not pass through raw."""
    model = MLP(in_channels=4, hidden_channels=[8, 1])
    optimizer = model.get_optimizer("AdamWScheduleFree", lr=1.0)
    assert isinstance(optimizer, AdamWScheduleFree)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0025)


def test_train_model_switches_schedulefree_modes():
    epochs = 3
    model = SpyModel()
    objective = TorchObjective(_mock_experiment(epochs=epochs))
    train_loader, val_loader = _loaders()
    params = {"lr": 1.0, "optimizer": "AdamWScheduleFree", "epochs": epochs}

    metrics = objective.train_model(model, train_loader, val_loader, params)

    optimizer = model.optimizer
    assert optimizer.train_calls == epochs
    # one eval per validation pass plus the trailing eval after the loop
    assert optimizer.eval_calls == epochs + 1
    assert all(g.get("train_mode", True) is False for g in optimizer.param_groups)
    assert np.isfinite(metrics["val_loss"])


def test_train_model_without_val_loader_ends_in_eval_mode():
    model = SpyModel()
    objective = TorchObjective(_mock_experiment(epochs=2))
    train_loader, _ = _loaders()
    params = {"lr": 1.0, "optimizer": "AdamWScheduleFree", "epochs": 2}

    metrics = objective.train_model(model, train_loader, None, params)

    optimizer = model.optimizer
    assert optimizer.eval_calls == 1
    assert all(g.get("train_mode", True) is False for g in optimizer.param_groups)
    assert np.isfinite(metrics["val_loss"])


def test_torch_objective_end_to_end_with_schedulefree():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 3)).astype(np.float32)
    y = rng.normal(size=(24, 1)).astype(np.float32)
    dataset = SpotDataFromArray(x_train=X, y_train=y, x_val=X, y_val=y)

    params = ParameterSet()
    params.add_float("lr", 0.5, 2.0, default=1.0)
    params.add_factor("optimizer", ["AdamWScheduleFree"], default="AdamWScheduleFree")

    experiment = ExperimentControl(
        dataset=dataset,
        model_class=MLP,
        hyperparameters=params,
        seed=1,
        loss_function=nn.MSELoss(),
        metrics=["val_loss"],
        epochs=2,
        batch_size=8,
    )
    objective = TorchObjective(experiment, seed=1)

    result = objective(np.array([[1.0, 0.0]]))

    assert result.shape == (1, 1)
    assert np.isfinite(result[0, 0])
