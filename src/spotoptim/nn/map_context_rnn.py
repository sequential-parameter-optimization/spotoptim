# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Hierarchical line-context RNN for whole compressor-map regression.

Extends the per-line `spotoptim.nn.many_to_many_rnn.ManyToManyRNN` with an
"orthogonal" second pass: an inner (optionally bidirectional) ``nn.RNN``
encodes each speed line point by point, and an outer ``nn.RNN`` runs across
the resulting line embeddings — in the order the lines appear in the map
tensor, e.g. ascending by the reduced-speed band — so every line receives a
context vector describing its neighboring lines. The context is concatenated
to each point state before the output head, letting predictions depend on the
whole map instead of a single line. Only input features flow through both
passes, so no target information leaks between lines.

One forward pass processes one map (a padded stack of its lines, e.g. from
`spotoptim.data.manydataset.load_map_data`); training over several maps is
provided by `spotoptim.nn.training.train_maps`.

Requires the ``torch`` optional extra (``pip install 'spotoptim[torch]'``).
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MapContextRNN(nn.Module):
    """Two-level recurrent network mapping a whole map to per-point outputs.

    The inner ``rnn_layer`` matches `ManyToManyRNN`: the padded lines are
    packed with their true lengths and encoded point by point. Its final
    hidden states (forward and backward concatenated) form one embedding per
    line; the outer ``context_layer`` runs over these embeddings in line
    order and yields a per-line context vector, which is broadcast to every
    point of its line. The head concatenates point state and line context and
    applies dropout, a fully connected layer, an activation, and a linear
    output layer.

    Args:
        input_size (int): Number of input features per point.
        output_size (int, optional): Number of outputs per point. Defaults to 1.
        rnn_units (int, optional): Hidden size of the inner (per-line) RNN.
            Defaults to 256.
        fc_units (int, optional): Width of the fully connected layer.
            Defaults to 256.
        context_units (int, optional): Hidden size of the outer (across-line)
            RNN. Defaults to 64.
        activation_fct (Optional[nn.Module], optional): Activation between the
            fully connected layer and the output head. Defaults to ``nn.ReLU()``.
        dropout (float, optional): Dropout probability applied to the
            concatenated point/context representation. Defaults to 0.0.
        bidirectional (bool, optional): Whether both RNNs are bidirectional.
            Defaults to True.

    Note:
        - One forward pass processes ONE map; the batch dimension of the
          input is the number of lines in that map.
        - The across-line order is the row order of the input tensor. With
          `load_map_data` this is ascending in the ``group_by`` column.

    Examples:
        ```{python}
        import torch
        from spotoptim.nn.map_context_rnn import MapContextRNN

        torch.manual_seed(0)
        model = MapContextRNN(input_size=2, rnn_units=8, fc_units=8, context_units=4)
        lines = torch.zeros(3, 5, 2)  # one map: 3 lines, up to 5 points
        lengths = torch.tensor([5, 3, 4])
        print(model(lines, lengths).shape)
        ```
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        rnn_units: int = 256,
        fc_units: int = 256,
        context_units: int = 64,
        activation_fct: Optional[nn.Module] = None,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ):
        super().__init__()
        if activation_fct is None:
            activation_fct = nn.ReLU()
        num_directions = 2 if bidirectional else 1
        self.rnn_layer = nn.RNN(
            input_size=input_size,
            hidden_size=rnn_units,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.context_layer = nn.RNN(
            input_size=rnn_units * num_directions,
            hidden_size=context_units,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(
            rnn_units * num_directions + context_units * num_directions, fc_units
        )
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(fc_units, output_size)
        self.activation_fct = activation_fct

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Compute per-point outputs for one padded map.

        Args:
            x (torch.Tensor): Padded lines of one map, shape
                ``(L, T_max, input_size)``, ordered along the across-line axis.
            lengths (torch.Tensor): True line lengths, shape ``(L,)``.

        Returns:
            torch.Tensor: Padded outputs, shape ``(L, T_max, output_size)``.
        """
        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, h_n = self.rnn_layer(packed)
        h, _ = pad_packed_sequence(packed_output, batch_first=True)
        # line embeddings from the final hidden states at each line's true
        # length: (num_directions, L, rnn_units) -> (L, num_directions*rnn_units)
        embeddings = h_n.transpose(0, 1).reshape(h.size(0), -1)
        # orthogonal pass across the lines of the map
        context, _ = self.context_layer(embeddings.unsqueeze(0))
        context = context.squeeze(0).unsqueeze(1).expand(-1, h.size(1), -1)
        z = torch.cat([h, context], dim=-1)
        z = self.dropout(z)
        z = self.fc(z)
        z = self.activation_fct(z)
        return self.output_layer(z)
