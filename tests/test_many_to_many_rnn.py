# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotoptim.nn.many_to_many_rnn and spotoptim.nn.optimizer."""

import pytest
import torch
import torch.nn as nn

from spotoptim.nn.many_to_many_rnn import (
    ManyToManyRNN,
    ManyToManyRNNRegressor,
    get_activation,
)
from spotoptim.nn.optimizer import optimizer_handler


def test_get_activation_matches_spotpython_conventions():
    assert isinstance(get_activation("Sigmoid"), nn.Sigmoid)
    assert isinstance(get_activation("Tanh"), nn.Tanh)
    assert isinstance(get_activation("ReLU"), nn.ReLU)
    leaky = get_activation("LeakyReLU")
    assert isinstance(leaky, nn.LeakyReLU)
    # spotPython's custom LeakyReLU uses alpha=0.1, not the torch default 0.01
    assert leaky.negative_slope == pytest.approx(0.1)
    assert isinstance(get_activation("ELU"), nn.ELU)
    # Swish(x) = x * sigmoid(x) = SiLU
    assert isinstance(get_activation("Swish"), nn.SiLU)


def test_get_activation_unknown_name_raises():
    with pytest.raises(ValueError, match="not supported"):
        get_activation("Bogus")


@pytest.mark.parametrize("bidirectional", [True, False])
def test_many_to_many_rnn_forward_shape(bidirectional):
    torch.manual_seed(0)
    model = ManyToManyRNN(
        input_size=1, rnn_units=8, fc_units=4, bidirectional=bidirectional
    )
    x = torch.randn(3, 5, 1)
    lengths = torch.tensor([5, 3, 2])
    out = model(x, lengths)
    assert out.shape == (3, 5, 1)


def test_regressor_accepts_hyperdict_names_and_extra_kwargs():
    torch.manual_seed(0)
    model = ManyToManyRNNRegressor(
        input_dim=1,
        output_dim=1,
        rnn_units=16,
        fc_units=8,
        act_fn="Swish",
        dropout_prob=0.1,
        bidirectional=True,
        # surplus tuning hyperparameters must be tolerated:
        epochs=128,
        batch_size=2,
        patience=16,
        optimizer="Adam",
        lr_mult=1.0,
    )
    x = torch.randn(2, 4, 1)
    lengths = torch.tensor([4, 2])
    assert model(x, lengths).shape == (2, 4, 1)
    assert isinstance(model.layers.activation_fct, nn.SiLU)


def test_optimizer_handler_learning_rate_conventions():
    params = [nn.Parameter(torch.zeros(3))]
    cases = {
        "Adadelta": 0.5 * 1.0,
        "Adagrad": 0.5 * 0.01,
        "Adam": 0.5 * 0.001,
        "AdamW": 0.5 * 0.001,
        "Adamax": 0.5 * 0.002,
        "ASGD": 0.5 * 0.01,
        "NAdam": 0.5 * 0.002,
        "RMSprop": 0.5 * 0.01,
        "Rprop": 0.5 * 0.01,
        "SGD": 0.5 * 1e-3,
        # lr_mult is ignored for these two, matching spotPython:
        "SparseAdam": 0.001,
        "RAdam": 0.001,
    }
    for name, expected_lr in cases.items():
        opt = optimizer_handler(name, params, lr_mult=0.5)
        assert opt.param_groups[0]["lr"] == pytest.approx(expected_lr), name


def test_optimizer_handler_unknown_name_raises():
    with pytest.raises(ValueError, match="not supported"):
        optimizer_handler("Bogus", [nn.Parameter(torch.zeros(1))])
