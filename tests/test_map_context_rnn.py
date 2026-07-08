# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for MapContextRNN, load_map_data, train_maps, and evaluate_map."""

import math

import pandas as pd
import pytest
import torch
import torch.nn as nn

from spotoptim.data.manydataset import load_map_data
from spotoptim.nn.many_to_many_rnn import ManyToManyRNN
from spotoptim.nn.map_context_rnn import MapContextRNN
from spotoptim.nn.training import evaluate_map, train_maps
from spotoptim.utils.seed import seed_everything


def _toy_map_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "line": [1, 1, 1, 2, 2, 3, 3, 3, 3],
            "u": [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0],
            "flow": [0.1, 0.2, 0.3, 0.15, 0.25, 0.1, 0.2, 0.3, 0.4],
            "y": [1.0, 2.0, 3.0, 1.5, 2.5, 1.2, 2.2, 3.2, 4.2],
        }
    )


def _small_model(**kwargs) -> MapContextRNN:
    defaults = dict(input_size=2, rnn_units=8, fc_units=8, context_units=4)
    defaults.update(kwargs)
    return MapContextRNN(**defaults)


class _ConstantModel(nn.Module):
    """Predicts 1.0 for every point; for hand-computed metric checks."""

    def forward(self, x, lengths):
        return torch.ones(x.shape[0], x.shape[1], 1)


def test_forward_shape_variable_lengths():
    seed_everything(0)
    model = _small_model()
    x = torch.randn(3, 5, 2)
    lengths = torch.tensor([5, 2, 4])
    out = model(x, lengths)
    assert out.shape == (3, 5, 1)
    assert torch.isfinite(out).all()


def test_forward_single_line_map():
    seed_everything(0)
    model = _small_model()
    out = model(torch.randn(1, 4, 2), torch.tensor([4]))
    assert out.shape == (1, 4, 1)


def test_context_carries_information_across_lines():
    """Changing another line changes this line's prediction (unlike ManyToManyRNN)."""
    seed_everything(0)
    model = _small_model()
    x = torch.randn(3, 5, 2)
    lengths = torch.tensor([5, 5, 5])
    x_perturbed = x.clone()
    x_perturbed[2] += 1.0

    out_line0 = model(x, lengths)[0]
    out_line0_perturbed = model(x_perturbed, lengths)[0]
    assert not torch.allclose(out_line0, out_line0_perturbed)

    seed_everything(0)
    baseline = ManyToManyRNN(input_size=2, rnn_units=8, fc_units=8)
    base_line0 = baseline(x, lengths)[0]
    base_line0_perturbed = baseline(x_perturbed, lengths)[0]
    assert torch.allclose(base_line0, base_line0_perturbed)


def test_forward_deterministic_with_seed():
    seed_everything(123)
    model_a = _small_model()
    seed_everything(123)
    model_b = _small_model()
    x = torch.zeros(2, 3, 2)
    lengths = torch.tensor([3, 2])
    assert torch.equal(model_a(x, lengths), model_b(x, lengths))


def test_load_map_data_shapes_and_order():
    x, lengths, y = load_map_data(
        _toy_map_df(), target="y", group_by="line", drop="line"
    )
    assert x.shape == (3, 4, 2)
    assert lengths.tolist() == [3, 2, 4]
    assert y.shape == (3, 4)
    # lines ordered ascending by the group_by column: u = 1, 2, 3
    assert x[0, 0, 0].item() == 1.0
    assert x[1, 0, 0].item() == 2.0
    assert x[2, 0, 0].item() == 3.0
    # padding is zero beyond the true length
    assert y[1, 2:].tolist() == [0.0, 0.0]


def test_train_maps_reduces_loss():
    seed_everything(42)
    maps = [load_map_data(_toy_map_df(), target="y", group_by="line", drop="line")]
    model = _small_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model, losses = train_maps(
        model, maps, optimizer, nn.MSELoss(), epochs=50, verbose=False
    )
    assert len(losses) == 50
    assert losses[-1] < losses[0]


def test_evaluate_map_structure_and_hand_computed_metrics():
    map_data = load_map_data(_toy_map_df(), target="y", group_by="line", drop="line")
    x_ls, y_hat_ls, y_ls, mape, rmse = evaluate_map(_ConstantModel(), map_data)
    assert len(x_ls) == len(y_hat_ls) == len(y_ls) == len(mape) == len(rmse) == 3
    assert [len(line) for line in y_hat_ls] == [3, 2, 4]
    # line 1: y = [1, 2, 3], y_hat = 1 -> MAPE = (0 + 1/2 + 2/3) / 3
    assert mape[0] == pytest.approx((0.0 + 0.5 + 2.0 / 3.0) / 3.0)
    # RMSE = sqrt((0 + 1 + 4) / 3)
    assert rmse[0] == pytest.approx(math.sqrt(5.0 / 3.0))

    mean_mape, mean_rmse = evaluate_map(_ConstantModel(), map_data, metrics_only=True)
    assert mean_mape == pytest.approx(sum(mape) / 3.0)
    assert mean_rmse == pytest.approx(sum(rmse) / 3.0)


def test_evaluate_map_excludes_padding():
    """Padded positions must not enter the per-line metrics."""
    map_data = load_map_data(_toy_map_df(), target="y", group_by="line", drop="line")
    _, _, y_ls, _, _ = evaluate_map(_ConstantModel(), map_data)
    # line 2 has true length 2: targets [1.5, 2.5], no padded zeros
    assert y_ls[1] == [1.5, 2.5]
