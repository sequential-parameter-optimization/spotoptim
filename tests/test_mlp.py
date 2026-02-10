# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch
import torch.nn as nn
from spotoptim.nn.mlp import MLP


def test_mlp_initialization():
    mlp = MLP(in_channels=10, hidden_channels=[20, 30])
    assert len(mlp) == 5  # (Linear, ReLU, Dropout) * 1 + (Linear, Dropout) * 1
    # hidden_channels[:-1] is [20]. Loop runs once.
    # Loop: Linear(10, 20), ReLU, Dropout. 3 layers.
    # Last: Linear(20, 30), Dropout. 2 layers.
    # Total 5 layers?
    # Let's check logic:
    # default activation is ReLU. default norm is None.
    # Loop:
    #  Linear
    #  norm (None)
    #  activation (ReLU)
    #  Dropout
    # Last:
    #  Linear
    #  Dropout

    # 3 layers in loop + 2 layers at end = 5 layers.

    # Let's count properly by checking types
    assert isinstance(mlp[0], nn.Linear)
    assert isinstance(mlp[1], nn.ReLU)
    assert isinstance(mlp[2], nn.Dropout)
    assert isinstance(mlp[3], nn.Linear)
    assert isinstance(mlp[4], nn.Dropout)

    # Check dimensions
    assert mlp[0].in_features == 10
    assert mlp[0].out_features == 20
    assert mlp[3].in_features == 20
    assert mlp[3].out_features == 30


def test_mlp_forward():
    in_channels = 5
    hidden_channels = [10, 2]  # 1 hidden layer of 10, output 2
    mlp = MLP(in_channels=in_channels, hidden_channels=hidden_channels)

    batch_size = 3
    x = torch.randn(batch_size, in_channels)
    out = mlp(x)

    assert out.shape == (batch_size, 2)


def test_mlp_with_norm_and_dropout():
    in_channels = 5
    hidden_channels = [10, 5, 2]
    # 2 hidden layers in loop (10, 5). Last layer (2).
    # hidden_channels[:-1] = [10, 5]

    mlp = MLP(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        norm_layer=nn.BatchNorm1d,
        dropout=0.5,
    )

    # Loop 1 (dest 10): Linear, BatchNorm, ReLU, Dropout
    # Loop 2 (dest 5): Linear, BatchNorm, ReLU, Dropout
    # Last (dest 2): Linear, Dropout

    assert len(mlp) == 4 + 4 + 2  # 10 layers

    assert isinstance(mlp[1], nn.BatchNorm1d)
    assert isinstance(mlp[3], nn.Dropout)
    assert mlp[3].p == 0.5

    # Check last dropout
    assert isinstance(mlp[-1], nn.Dropout)
    assert mlp[-1].p == 0.5


def test_mlp_no_activation():
    mlp = MLP(in_channels=5, hidden_channels=[10], activation_layer=None)
    # Loop empty (hidden_channels[:-1] is [])
    # Last: Linear, Dropout

    assert len(mlp) == 2
    assert isinstance(mlp[0], nn.Linear)
    assert isinstance(mlp[1], nn.Dropout)


def test_mlp_inplace():
    mlp = MLP(in_channels=5, hidden_channels=[10, 10], inplace=True)
    # Loop 1: Linear, ReLU(inplace=True), Dropout(inplace=True)
    assert mlp[1].inplace is True
    assert mlp[2].inplace is True  # Dropout supports inplace


def test_mlp_bias():
    mlp_bias = MLP(in_channels=5, hidden_channels=[10], bias=True)
    assert mlp_bias[0].bias is not None

    mlp_no_bias = MLP(in_channels=5, hidden_channels=[10], bias=False)
    assert mlp_no_bias[0].bias is None


def test_mlp_lr():
    # Test default lr
    mlp = MLP(in_channels=10, hidden_channels=[20])
    assert mlp.lr == 1.0

    # Test custom lr
    mlp_custom = MLP(in_channels=10, hidden_channels=[20], lr=0.1)
    assert mlp_custom.lr == 0.1


def test_get_optimizer():
    mlp = MLP(in_channels=10, hidden_channels=[20], lr=0.5)

    # Test Adam creation with unified lr
    # 0.5 * 0.001 (standard Adam default) = 0.0005
    opt = mlp.get_optimizer("Adam")
    assert isinstance(opt, torch.optim.Adam)
    assert opt.defaults["lr"] == 0.0005

    # Test overridden lr
    # 2.0 * 0.001 = 0.002
    opt_override = mlp.get_optimizer("Adam", lr=2.0)
    assert opt_override.defaults["lr"] == 0.002

    # Test SGD
    # 0.5 * 0.01 (standard SGD default) = 0.005
    opt_sgd = mlp.get_optimizer("SGD", momentum=0.9)
    assert isinstance(opt_sgd, torch.optim.SGD)
    assert opt_sgd.defaults["lr"] == 0.005
    assert opt_sgd.defaults["momentum"] == 0.9


def test_get_default_parameters():
    from spotoptim.hyperparameters.parameters import ParameterSet

    params = MLP.get_default_parameters()
    assert isinstance(params, ParameterSet)

    names = params.names()
    expected = ["l1", "num_hidden_layers", "activation", "lr", "optimizer"]
    for name in expected:
        assert name in names


def test_mlp_alternative_init():
    # Test initialization with l1, num_hidden_layers, output_dim
    input_dim = 10
    output_dim = 5
    l1 = 20
    num_hidden_layers = 3

    # Implicit hidden_channels should be [20, 20, 20, 5]
    # Structure:
    # Layer 1: 10 -> 20 (Linear, ReLU, Dropout)
    # Layer 2: 20 -> 20 (Linear, ReLU, Dropout)
    # Layer 3: 20 -> 20 (Linear, ReLU, Dropout)
    # Layer 4: 20 -> 5  (Linear, Dropout)

    model = MLP(
        in_channels=input_dim,
        hidden_channels=None,  # Explicitly None to trigger alternative logic
        l1=l1,
        num_hidden_layers=num_hidden_layers,
        output_dim=output_dim,
    )

    # Total layers: (3 hidden * 3 ops) + (1 output * 2 ops) = 9 + 2 = 11?
    # Loop over hidden_channels[:-1]: [20, 20, 20]. 3 iterations.
    # Each iter: Linear, ReLU, Dropout.
    # Last: Linear(20, 5), Dropout.
    # Total layers = 3*3 + 2 = 11.

    assert len(model) == 11

    # Check dimensions
    # First layer
    assert model[0].in_features == 10
    assert model[0].out_features == 20

    # Last linear layer (second to last op)
    assert isinstance(model[-2], nn.Linear)
    assert model[-2].out_features == 5


def test_mlp_priority():
    # hidden_channels should take precedence over l1/num_hidden_layers

    model = MLP(
        in_channels=10,
        hidden_channels=[30, 1],
        l1=100,
        num_hidden_layers=10,
        output_dim=50,
    )

    # Should use [30, 1] ->
    # Loop [30]: Linear(10, 30), ReLU, Dropout.
    # Last: Linear(30, 1), Dropout.
    # Total 5 layers.

    assert len(model) == 5
    assert model[0].out_features == 30
    assert model[-2].out_features == 1
