# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Variable-length sequence datasets and padding collates for recurrent models.

Ported from spotPython (``spotpython.data.manydataset`` and the padding
collates from ``spotpython.data.lightdatamodule``) without any Lightning
dependency. Each dataset item is one complete sequence (e.g. one operating
curve of a compressor map), so batches must be padded with the collate
classes provided here.

Requires the ``torch`` optional extra (``pip install 'spotoptim[torch]'``).
"""

from typing import List, Optional, Tuple, Union

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, Dataset


class ManyToManyDataset(Dataset):
    """Sequence dataset with one target value per time step.

    Each element of ``df_list`` is one variable-length sequence; item ``i``
    is the pair ``(features_i, targets_i)`` with shapes ``(T_i, n_features)``
    and ``(T_i,)``.

    Args:
        df_list (List[pd.DataFrame]): List of pandas DataFrames, one per sequence.
        target (str): The target column name.
        drop (Optional[Union[str, List[str]]]): Column(s) to drop from the
            DataFrames before extracting features. If a listed column is
            missing, no column is dropped. Defaults to None.
        dtype (torch.dtype): Data type for the tensors. Defaults to ``torch.float32``.

    Attributes:
        data (List[pd.DataFrame]): DataFrames with the ``drop`` columns removed.
        target (List[torch.Tensor]): Per-sequence target tensors, shape ``(T_i,)``.
        features (List[torch.Tensor]): Per-sequence feature tensors, shape
            ``(T_i, n_features)``.

    Examples:
        ```{python}
        import pandas as pd
        from spotoptim.data.manydataset import ManyToManyDataset

        df1 = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
        df2 = pd.DataFrame({"x": [4.0, 5.0], "y": [8.0, 10.0]})
        ds = ManyToManyDataset([df1, df2], target="y")
        print(len(ds))
        features, targets = ds[0]
        print(features.shape, targets.shape)
        ```
    """

    def __init__(
        self,
        df_list: List[pd.DataFrame],
        target: str,
        drop: Optional[Union[str, List[str]]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        if drop is None:
            self.data = list(df_list)
        else:
            try:
                self.data = [df.drop(drop, axis=1) for df in df_list]
            except KeyError:
                self.data = df_list
        self.target = [
            torch.tensor(df[target].to_numpy(), dtype=dtype) for df in self.data
        ]
        self.features = [
            torch.tensor(df.drop([target], axis=1).to_numpy(), dtype=dtype)
            for df in self.data
        ]

    def __getitem__(self, index: int):
        """Return the ``(features, targets)`` pair of sequence ``index``."""
        x = self.features[index]
        y = self.target[index]
        return x, y

    def __len__(self) -> int:
        """Return the number of sequences."""
        return len(self.data)


class ManyToOneDataset(Dataset):
    """Sequence dataset with a single target value per sequence.

    Like `ManyToManyDataset`, but item ``i`` pairs the full feature sequence
    with the scalar target taken from the first row of sequence ``i``.

    Args:
        df_list (List[pd.DataFrame]): List of pandas DataFrames, one per sequence.
        target (str): The target column name.
        drop (Optional[Union[str, List[str]]]): Column(s) to drop from the
            DataFrames before extracting features. If a listed column is
            missing, no column is dropped. Defaults to None.
        dtype (torch.dtype): Data type for the tensors. Defaults to ``torch.float32``.

    Attributes:
        data (List[pd.DataFrame]): DataFrames with the ``drop`` columns removed.
        target (List[torch.Tensor]): Per-sequence scalar target tensors.
        features (List[torch.Tensor]): Per-sequence feature tensors, shape
            ``(T_i, n_features)``.

    Examples:
        ```{python}
        import pandas as pd
        from spotoptim.data.manydataset import ManyToOneDataset

        df1 = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [5.0, 5.0, 5.0]})
        df2 = pd.DataFrame({"x": [4.0, 5.0], "y": [7.0, 7.0]})
        ds = ManyToOneDataset([df1, df2], target="y")
        features, target = ds[1]
        print(features.shape, target)
        ```
    """

    def __init__(
        self,
        df_list: List[pd.DataFrame],
        target: str,
        drop: Optional[Union[str, List[str]]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        if drop is None:
            self.data = list(df_list)
        else:
            try:
                self.data = [df.drop(drop, axis=1) for df in df_list]
            except KeyError:
                self.data = df_list
        self.target = [
            torch.tensor(df[target].to_numpy()[0], dtype=dtype) for df in self.data
        ]
        self.features = [
            torch.tensor(df.drop([target], axis=1).to_numpy(), dtype=dtype)
            for df in self.data
        ]

    def __getitem__(self, index: int):
        """Return the ``(features, target)`` pair of sequence ``index``."""
        x = self.features[index]
        y = self.target[index]
        return x, y

    def __len__(self) -> int:
        """Return the number of sequences."""
        return len(self.data)


class PadSequenceManyToMany:
    """Padding collate for `ManyToManyDataset` batches.

    Pads features and targets of a batch of variable-length sequences with
    zeros to the longest sequence in the batch and records the true lengths,
    as required by ``torch.nn.utils.rnn.pack_padded_sequence``.

    Examples:
        ```{python}
        import pandas as pd
        from torch.utils.data import DataLoader
        from spotoptim.data.manydataset import ManyToManyDataset, PadSequenceManyToMany

        df1 = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
        df2 = pd.DataFrame({"x": [4.0, 5.0], "y": [8.0, 10.0]})
        ds = ManyToManyDataset([df1, df2], target="y")
        dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=PadSequenceManyToMany())
        x, lengths, y = next(iter(dl))
        print(x.shape, lengths.tolist(), y.shape)
        ```
    """

    def __call__(self, batch):
        """Collate a batch into ``(padded_x, lengths, padded_y)``.

        Args:
            batch: Sequence of ``(features, targets)`` pairs as produced by
                `ManyToManyDataset`.

        Returns:
            tuple: ``(padded_x, lengths, padded_y)`` where ``padded_x`` has
                shape ``(B, T_max, n_features)``, ``lengths`` is an int tensor of
                shape ``(B,)``, and ``padded_y`` has shape ``(B, T_max)``.
        """
        batch_x, batch_y = zip(*batch)
        padded_batch_x = pad_sequence(list(batch_x), batch_first=True)
        padded_batch_y = pad_sequence(list(batch_y), batch_first=True)
        lengths = torch.tensor([len(x) for x in batch_x])

        return padded_batch_x, lengths, padded_batch_y


class PadSequenceManyToOne:
    """Padding collate for `ManyToOneDataset` batches.

    Pads the feature sequences with zeros and stacks the scalar targets.

    Examples:
        ```{python}
        import pandas as pd
        from torch.utils.data import DataLoader
        from spotoptim.data.manydataset import ManyToOneDataset, PadSequenceManyToOne

        df1 = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [5.0, 5.0, 5.0]})
        df2 = pd.DataFrame({"x": [4.0, 5.0], "y": [7.0, 7.0]})
        ds = ManyToOneDataset([df1, df2], target="y")
        dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=PadSequenceManyToOne())
        x, lengths, y = next(iter(dl))
        print(x.shape, lengths.tolist(), y.shape)
        ```
    """

    def __call__(self, batch):
        """Collate a batch into ``(padded_x, lengths, y)``.

        Args:
            batch: Sequence of ``(features, target)`` pairs as produced by
                `ManyToOneDataset`.

        Returns:
            tuple: ``(padded_x, lengths, y)`` where ``padded_x`` has shape
                ``(B, T_max, n_features)``, ``lengths`` is an int tensor of shape
                ``(B,)``, and ``y`` is a tensor of shape ``(B,)``.
        """
        batch_x, batch_y = zip(*batch)
        padded_batch_x = pad_sequence(list(batch_x), batch_first=True)
        lengths = torch.tensor([len(x) for x in batch_x])

        return padded_batch_x, lengths, torch.tensor(batch_y)


def load_sequence_data(
    data: pd.DataFrame,
    target: str,
    group_by: str,
    drop: Optional[Union[str, List[str]]] = None,
    input_features: Optional[List[str]] = None,
    feature_scaling=None,
    target_scaling=None,
    dataset_type: str = "many_to_many",
) -> Tuple[Dataset, pd.DataFrame]:
    """Group a DataFrame into a variable-length sequence dataset.

    Splits ``data`` by the values of ``group_by`` (e.g. one compressor speed
    line per group) and wraps the groups in a `ManyToManyDataset` or
    `ManyToOneDataset`. Ported from the schu25a study's ``load_data`` helper
    (``src/rnn/utils.py``); the experiment-specific column defaults were
    removed.

    Args:
        data (pd.DataFrame): Input data; one row per time step.
        target (str): The target column name.
        group_by (str): Column whose values define the sequences.
        drop (Optional[Union[str, List[str]]]): Column(s) to drop from the
            groups before extracting features. Defaults to None.
        input_features (Optional[List[str]]): Columns scaled by
            ``feature_scaling``. Only used when ``feature_scaling`` is given;
            the dataset features are always all columns except ``drop`` and
            ``target``. Defaults to None.
        feature_scaling: Optional sklearn-style scaler; applied in place to
            ``data[input_features]`` via ``fit_transform``. Defaults to None.
        target_scaling: Optional sklearn-style scaler; applied in place to
            ``data[target]`` via ``fit_transform``. Defaults to None.
        dataset_type (str, optional): Dataset flavor. Options:
            - "many_to_many": one target value per time step.
            - "many_to_one": one scalar target per sequence.
            Defaults to "many_to_many".

    Returns:
        tuple: ``(dataset, data)`` — the sequence dataset and the (possibly
            scaled) DataFrame.

    Raises:
        ValueError: If ``dataset_type`` is not "many_to_many" or "many_to_one".

    Examples:
        ```{python}
        import pandas as pd
        from spotoptim.data.manydataset import load_sequence_data

        df = pd.DataFrame({
            "line": [1, 1, 1, 2, 2],
            "x": [0.1, 0.2, 0.3, 0.4, 0.5],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        ds, df = load_sequence_data(df, target="y", group_by="line", drop="line")
        print(len(ds), ds[0][0].shape)
        ```
    """
    if feature_scaling is not None:
        data[input_features] = feature_scaling.fit_transform(data[input_features])

    if target_scaling is not None:
        data[target] = target_scaling.fit_transform(data[target])

    groups = [group for _, group in data.groupby(group_by)]

    if dataset_type == "many_to_many":
        return ManyToManyDataset(groups, target=target, drop=drop), data
    elif dataset_type == "many_to_one":
        return ManyToOneDataset(groups, target=target, drop=drop), data
    raise ValueError(
        f"dataset_type {dataset_type!r} not supported. "
        "Options: 'many_to_many', 'many_to_one'."
    )


def load_pooled_sequence_data(
    data_list: List[pd.DataFrame],
    target: str,
    group_by: str,
    drop: Optional[Union[str, List[str]]] = None,
    input_features: Optional[List[str]] = None,
    feature_scaling=None,
    target_scaling=None,
    dataset_type: str = "many_to_many",
) -> ConcatDataset:
    """Pool several DataFrames into one concatenated sequence dataset.

    Applies `load_sequence_data` to every DataFrame (e.g. one compressor map
    each) and concatenates the resulting sequence datasets — the pooled
    training set of the schu25a "global training" workflow (ported from
    ``load_pretrain_data`` in ``src/rnn/utils.py``).

    Args:
        data_list (List[pd.DataFrame]): Input DataFrames; one per map.
        target (str): The target column name.
        group_by (str): Column whose values define the sequences.
        drop (Optional[Union[str, List[str]]]): Column(s) to drop from the
            groups before extracting features. Defaults to None.
        input_features (Optional[List[str]]): Columns scaled by
            ``feature_scaling``; see `load_sequence_data`. Defaults to None.
        feature_scaling: Optional sklearn-style scaler, applied per DataFrame.
            Defaults to None.
        target_scaling: Optional sklearn-style scaler, applied per DataFrame.
            Defaults to None.
        dataset_type (str, optional): "many_to_many" or "many_to_one".
            Defaults to "many_to_many".

    Returns:
        ConcatDataset: The concatenation of the per-DataFrame sequence datasets.

    Examples:
        ```{python}
        import pandas as pd
        from spotoptim.data.manydataset import load_pooled_sequence_data

        df1 = pd.DataFrame({"line": [1, 1, 2], "x": [0.1, 0.2, 0.3], "y": [1.0, 2.0, 3.0]})
        df2 = pd.DataFrame({"line": [1, 1], "x": [0.4, 0.5], "y": [4.0, 5.0]})
        pooled = load_pooled_sequence_data([df1, df2], target="y", group_by="line", drop="line")
        print(len(pooled))
        ```
    """
    ds_ls = []
    for data in data_list:
        ds, _ = load_sequence_data(
            data,
            target=target,
            group_by=group_by,
            drop=drop,
            input_features=input_features,
            feature_scaling=feature_scaling,
            target_scaling=target_scaling,
            dataset_type=dataset_type,
        )
        ds_ls.append(ds)

    return ConcatDataset(ds_ls)
