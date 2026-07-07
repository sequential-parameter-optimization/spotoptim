# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotoptim.nn.training, data grouping helpers, and utils.seed."""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spotoptim.data.manydataset import (
    PadSequenceManyToMany,
    load_pooled_sequence_data,
    load_sequence_data,
)
from spotoptim.nn.many_to_many_rnn import ManyToManyRNN
from spotoptim.nn.training import evaluate_sequences, train_sequences
from spotoptim.utils.seed import seed_everything


def _frame(lines, offset=0.0):
    rows = []
    for line, n in lines:
        for i in range(n):
            rows.append(
                {"line": line, "x": offset + 0.1 * (i + 1), "y": offset + float(i + 1)}
            )
    return pd.DataFrame(rows)


def test_load_sequence_data_groups_by_column():
    df = _frame([(1, 3), (2, 2)])
    ds, df_out = load_sequence_data(df, target="y", group_by="line", drop="line")
    assert len(ds) == 2
    x0, y0 = ds[0]
    assert x0.shape == (3, 1)  # only 'x' remains as feature
    assert y0.shape == (3,)
    assert df_out is df


def test_load_sequence_data_many_to_one_and_bad_type():
    df = _frame([(1, 3), (2, 2)])
    ds, _ = load_sequence_data(
        df, target="y", group_by="line", drop="line", dataset_type="many_to_one"
    )
    assert ds[0][1].dim() == 0
    with pytest.raises(ValueError, match="not supported"):
        load_sequence_data(df, target="y", group_by="line", dataset_type="bogus")


def test_load_pooled_sequence_data_concatenates():
    df1 = _frame([(1, 3), (2, 2)])
    df2 = _frame([(1, 4)], offset=10.0)
    pooled = load_pooled_sequence_data(
        [df1, df2], target="y", group_by="line", drop="line"
    )
    assert len(pooled) == 3  # 2 sequences + 1 sequence


def test_train_sequences_returns_model_and_history():
    seed_everything(0)
    df = _frame([(1, 3), (2, 4), (3, 2)])
    ds, _ = load_sequence_data(df, target="y", group_by="line", drop="line")
    dl = DataLoader(ds, batch_size=3, shuffle=False, collate_fn=PadSequenceManyToMany())
    model = ManyToManyRNN(input_size=1, rnn_units=4, fc_units=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model, losses = train_sequences(
        model, dl, optimizer, nn.MSELoss(), epochs=3, verbose=False
    )
    assert len(losses) == 3
    assert all(np.isfinite(v) for v in losses)


def test_train_sequences_with_val_loader_returns_val_history():
    seed_everything(0)
    df = _frame([(1, 3), (2, 4)])
    ds, _ = load_sequence_data(df, target="y", group_by="line", drop="line")
    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=PadSequenceManyToMany())
    val_dl = DataLoader(
        ds, batch_size=1, shuffle=False, collate_fn=PadSequenceManyToMany()
    )
    model = ManyToManyRNN(input_size=1, rnn_units=4, fc_units=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model, losses, val_losses = train_sequences(
        model, dl, optimizer, nn.MSELoss(), val_loader=val_dl, epochs=2, verbose=False
    )
    assert len(losses) == 2
    assert len(val_losses) == 2


def test_evaluate_sequences_matches_hand_computed_metrics():
    # Identity-free check: a constant-output model against known targets.
    class ConstModel(nn.Module):
        def forward(self, x, lengths):
            return torch.full((x.shape[0], x.shape[1], 1), 2.0)

    df = _frame([(1, 2)])  # targets [1.0, 2.0]
    ds, _ = load_sequence_data(df, target="y", group_by="line", drop="line")
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=PadSequenceManyToMany())
    x, y_hat, y, mape, rmse = evaluate_sequences(ConstModel(), dl)
    # y_hat = [2, 2] vs y = [1, 2]: MAPE = mean(|1-2|/1, |2-2|/2) = 0.5
    assert mape[0] == pytest.approx(0.5)
    # RMSE = sqrt(mean((1)^2, 0^2)) = sqrt(0.5)
    assert rmse[0] == pytest.approx(float(np.sqrt(0.5)))
    assert y[0] == [1.0, 2.0]
    metrics = evaluate_sequences(ConstModel(), dl, metrics_only=True)
    assert metrics == (pytest.approx(0.5), pytest.approx(float(np.sqrt(0.5))))


def test_seed_everything_makes_training_bit_identical():
    def run():
        seed_everything(123)
        df = _frame([(1, 3), (2, 4), (3, 2)])
        ds, _ = load_sequence_data(df, target="y", group_by="line", drop="line")
        dl = DataLoader(
            ds, batch_size=3, shuffle=True, collate_fn=PadSequenceManyToMany()
        )
        model = ManyToManyRNN(input_size=1, rnn_units=4, fc_units=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        _, losses = train_sequences(
            model, dl, optimizer, nn.MSELoss(), epochs=3, verbose=False
        )
        return losses

    assert run() == run()
