# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Many-to-many recurrent network for variable-length sequence regression.

Ported from ``spotpython.light.regression.ManyToManyRNNRegressor`` without the
Lightning wrapper: the architecture (packed bidirectional ``nn.RNN`` followed
by dropout, a fully connected layer, an activation, and a linear output head)
and its hyperparameter names are identical, so tuning results remain
comparable. Training is driven by an objective such as
`spotoptim.function.sequence_cv_objective.SequenceCVObjective` instead of a
Lightning trainer.

Requires the ``torch`` optional extra (``pip install 'spotoptim[torch]'``).
"""

from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#: Activation names of the spotPython ``light_hyper_dict`` ``act_fn`` factor,
#: mapped to mathematically equivalent torch modules. spotPython's custom
#: ``LeakyReLU`` uses ``alpha=0.1`` (not the torch default 0.01) and ``Swish``
#: equals ``nn.SiLU``.
_ACTIVATIONS = {
    "Sigmoid": lambda: nn.Sigmoid(),
    "Tanh": lambda: nn.Tanh(),
    "ReLU": lambda: nn.ReLU(),
    "LeakyReLU": lambda: nn.LeakyReLU(negative_slope=0.1),
    "ELU": lambda: nn.ELU(),
    "Swish": lambda: nn.SiLU(),
}


def get_activation(name: str) -> nn.Module:
    """Return the activation module for a spotPython ``act_fn`` level name.

    Args:
        name (str): Activation name. Options:
            - "Sigmoid"
            - "Tanh"
            - "ReLU"
            - "LeakyReLU": negative slope 0.1, matching spotPython.
            - "ELU"
            - "Swish": implemented as ``nn.SiLU``.

    Returns:
        nn.Module: A fresh activation module instance.

    Raises:
        ValueError: If ``name`` is not a supported activation name.

    Examples:
        ```{python}
        from spotoptim.nn.many_to_many_rnn import get_activation

        act = get_activation("LeakyReLU")
        print(type(act).__name__, act.negative_slope)
        ```
    """
    try:
        return _ACTIVATIONS[name]()
    except KeyError:
        raise ValueError(
            f"Activation {name!r} not supported. " f"Options: {sorted(_ACTIVATIONS)}"
        ) from None


class ManyToManyRNN(nn.Module):
    """Recurrent network mapping a padded sequence batch to per-step outputs.

    The input batch is packed with the true sequence lengths, passed through a
    single (optionally bidirectional) ``nn.RNN`` layer, unpacked, and fed
    through dropout, a fully connected layer, an activation, and a linear
    output head. Layer names and forward semantics match spotPython's
    ``ManyToManyRNN``.

    Args:
        input_size (int): Number of input features per time step.
        output_size (int, optional): Number of outputs per time step. Defaults to 1.
        rnn_units (int, optional): Hidden size of the RNN layer. Defaults to 256.
        fc_units (int, optional): Width of the fully connected layer. Defaults to 256.
        activation_fct (Optional[nn.Module], optional): Activation between the
            fully connected layer and the output head. Defaults to ``nn.ReLU()``.
        dropout (float, optional): Dropout probability applied to the RNN
            output. Defaults to 0.0.
        bidirectional (bool, optional): Whether the RNN is bidirectional.
            Defaults to True.

    Examples:
        ```{python}
        import torch
        from spotoptim.nn.many_to_many_rnn import ManyToManyRNN

        torch.manual_seed(0)
        model = ManyToManyRNN(input_size=1, rnn_units=8, fc_units=8)
        x = torch.zeros(2, 5, 1)
        lengths = torch.tensor([5, 3])
        print(model(x, lengths).shape)
        ```
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        rnn_units: int = 256,
        fc_units: int = 256,
        activation_fct: Optional[nn.Module] = None,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ):
        super().__init__()
        if activation_fct is None:
            activation_fct = nn.ReLU()
        self.rnn_layer = nn.RNN(
            input_size=input_size,
            hidden_size=rnn_units,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        if bidirectional:
            rnn_units = rnn_units * 2
        self.fc = nn.Linear(rnn_units, fc_units)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(fc_units, output_size)
        self.activation_fct = activation_fct

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Compute per-step outputs for a padded batch.

        Args:
            x (torch.Tensor): Padded input batch, shape ``(B, T_max, input_size)``.
            lengths (torch.Tensor): True sequence lengths, shape ``(B,)``.

        Returns:
            torch.Tensor: Padded outputs, shape ``(B, T_max, output_size)``.
        """
        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.rnn_layer(packed)
        x, _ = pad_packed_sequence(packed_output, batch_first=True)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.activation_fct(x)
        x = self.output_layer(x)
        return x


class ManyToManyRNNRegressor(nn.Module):
    """`ManyToManyRNN` with the spotPython hyperparameter interface.

    Accepts the hyperparameter names of the spotPython ``light_hyper_dict``
    entry ``ManyToManyRNNRegressor`` (``rnn_units``, ``fc_units``, ``act_fn``,
    ``dropout_prob``, ``bidirectional``) plus the ``input_dim``/``output_dim``
    convention used by spotoptim objectives. Training hyperparameters such as
    ``epochs``, ``batch_size``, ``patience``, ``optimizer``, and ``lr_mult``
    are accepted via ``**kwargs`` and ignored here — they are consumed by the
    training objective.

    Args:
        input_dim (int, optional): Number of input features per time step.
            Defaults to 1.
        output_dim (int, optional): Number of outputs per time step. Defaults to 1.
        rnn_units (int, optional): Hidden size of the RNN layer. Defaults to 256.
        fc_units (int, optional): Width of the fully connected layer. Defaults to 256.
        act_fn (Union[str, nn.Module], optional): Activation between the fully
            connected layer and the output head, either a name accepted by
            `get_activation` or a module instance. Defaults to "ReLU".
        dropout_prob (float, optional): Dropout probability applied to the RNN
            output. Defaults to 0.0.
        bidirectional (bool, optional): Whether the RNN is bidirectional.
            Defaults to True.
        **kwargs: Ignored. Accepts surplus tuning hyperparameters.

    Attributes:
        layers (ManyToManyRNN): The underlying recurrent network.

    Examples:
        ```{python}
        import torch
        from spotoptim.nn.many_to_many_rnn import ManyToManyRNNRegressor

        torch.manual_seed(0)
        model = ManyToManyRNNRegressor(
            input_dim=1, output_dim=1, rnn_units=16, fc_units=16,
            act_fn="Tanh", dropout_prob=0.1, epochs=128, batch_size=2,
        )
        x = torch.zeros(3, 4, 1)
        lengths = torch.tensor([4, 2, 3])
        print(model(x, lengths).shape)
        ```
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        rnn_units: int = 256,
        fc_units: int = 256,
        act_fn: Union[str, nn.Module] = "ReLU",
        dropout_prob: float = 0.0,
        bidirectional: bool = True,
        **kwargs,
    ):
        super().__init__()
        if isinstance(act_fn, str):
            act_fn = get_activation(act_fn)
        self.layers = ManyToManyRNN(
            input_size=input_dim,
            output_size=output_dim,
            rnn_units=rnn_units,
            fc_units=fc_units,
            activation_fct=act_fn,
            dropout=dropout_prob,
            bidirectional=bidirectional,
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Compute per-step outputs for a padded batch.

        Args:
            x (torch.Tensor): Padded input batch, shape ``(B, T_max, input_dim)``.
            lengths (torch.Tensor): True sequence lengths, shape ``(B,)``.

        Returns:
            torch.Tensor: Padded outputs, shape ``(B, T_max, output_dim)``.
        """
        return self.layers(x, lengths)
