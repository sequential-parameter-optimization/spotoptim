# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotoptim.data.manydataset (datasets and padding collates)."""

import pandas as pd
import torch
from torch.utils.data import DataLoader

from spotoptim.data.manydataset import (
    ManyToManyDataset,
    ManyToOneDataset,
    PadSequenceManyToMany,
    PadSequenceManyToOne,
)


def _frames():
    df1 = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0], "aux": [0.0, 0.0, 0.0]}
    )
    df2 = pd.DataFrame({"x": [4.0, 5.0], "y": [8.0, 10.0], "aux": [0.0, 0.0]})
    return [df1, df2]


def test_many_to_many_dataset_shapes_and_drop():
    ds = ManyToManyDataset(_frames(), target="y", drop="aux")
    assert len(ds) == 2
    x0, y0 = ds[0]
    assert x0.shape == (3, 1)
    assert y0.shape == (3,)
    assert x0.dtype == torch.float32
    x1, y1 = ds[1]
    assert x1.shape == (2, 1)
    assert torch.equal(y1, torch.tensor([8.0, 10.0]))


def test_many_to_many_dataset_missing_drop_column_keeps_frames():
    ds = ManyToManyDataset(_frames(), target="y", drop="does_not_exist")
    x0, _ = ds[0]
    # 'aux' not dropped, so two feature columns remain
    assert x0.shape == (3, 2)


def test_many_to_one_dataset_scalar_target():
    ds = ManyToOneDataset(_frames(), target="y", drop="aux")
    x1, y1 = ds[1]
    assert x1.shape == (2, 1)
    assert y1.dim() == 0
    assert y1.item() == 8.0


def test_pad_sequence_many_to_many_pads_with_zeros():
    ds = ManyToManyDataset(_frames(), target="y", drop="aux")
    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=PadSequenceManyToMany())
    x, lengths, y = next(iter(dl))
    assert x.shape == (2, 3, 1)
    assert y.shape == (2, 3)
    assert lengths.tolist() == [3, 2]
    # padded positions are zero
    assert x[1, 2, 0].item() == 0.0
    assert y[1, 2].item() == 0.0


def test_pad_sequence_many_to_one_stacks_targets():
    ds = ManyToOneDataset(_frames(), target="y", drop="aux")
    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=PadSequenceManyToOne())
    x, lengths, y = next(iter(dl))
    assert x.shape == (2, 3, 1)
    assert lengths.tolist() == [3, 2]
    assert y.tolist() == [2.0, 8.0]
