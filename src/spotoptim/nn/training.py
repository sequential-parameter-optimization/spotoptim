# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain training and evaluation loops for variable-length sequence models.

Ported op-for-op from the ``train``/``evaluate_many`` helpers of the schu25a
compressor-map study (``src/rnn/train.py``), so that runs seeded with
`spotoptim.utils.seed.seed_everything` reproduce the original results
bit-identically on the same device. The models are called as
``model(x, lengths)`` on padded batches (e.g.
`spotoptim.nn.many_to_many_rnn.ManyToManyRNN` with batches collated by
`spotoptim.data.manydataset.PadSequenceManyToMany`); losses and metrics are
computed on the squeezed padded tensors, exactly as in the original. The
original's unused ``seed`` parameters were dropped.

Requires the ``torch`` optional extra (``pip install 'spotoptim[torch]'``),
which includes ``torchmetrics``.
"""

from typing import List, Literal, Optional, Tuple, Union, overload

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.regression import MeanAbsolutePercentageError, MeanSquaredError


def train_sequences(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    device: str = "cpu",
    verbose: bool = True,
) -> Union[Tuple[nn.Module, List[float]], Tuple[nn.Module, List[float], List[float]]]:
    """Train a sequence model with a plain epoch loop.

    One optimizer step per batch; the loss is computed between
    ``model(x, lengths).squeeze()`` and the (padded) target batch. The
    recorded training loss is the batch-mean loss per epoch.

    Args:
        model (nn.Module): Model called as ``model(x, lengths)``.
        train_loader (DataLoader): Loader yielding ``(x, lengths, y)`` batches,
            e.g. collated by `PadSequenceManyToMany`.
        optimizer (torch.optim.Optimizer): Optimizer over ``model.parameters()``.
        criterion (nn.Module): Loss module, e.g. ``nn.MSELoss()``.
        val_loader (Optional[DataLoader]): Optional validation loader; when
            given, the mean per-batch RMSE from `evaluate_sequences` is
            recorded after every epoch. Defaults to None.
        epochs (int, optional): Number of epochs. Defaults to 10.
        device (str, optional): Training device. Defaults to "cpu".
        verbose (bool, optional): Print per-epoch losses. Defaults to True.

    Returns:
        tuple: ``(model, train_loss_ls)`` without a validation loader, or
            ``(model, train_loss_ls, val_loss_ls)`` with one.

    Examples:
        ```{python}
        import pandas as pd
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from spotoptim.data.manydataset import ManyToManyDataset, PadSequenceManyToMany
        from spotoptim.nn.many_to_many_rnn import ManyToManyRNN
        from spotoptim.nn.training import train_sequences
        from spotoptim.utils.seed import seed_everything

        seed_everything(42)
        frames = [
            pd.DataFrame({"x": [0.1, 0.2, 0.3], "y": [1.0, 2.0, 3.0]}),
            pd.DataFrame({"x": [0.4, 0.5], "y": [4.0, 5.0]}),
        ]
        ds = ManyToManyDataset(frames, target="y")
        dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=PadSequenceManyToMany())
        model = ManyToManyRNN(input_size=1, rnn_units=8, fc_units=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model, losses = train_sequences(
            model, dl, optimizer, nn.MSELoss(), epochs=2, verbose=False
        )
        print(len(losses))
        ```
    """
    model.train()
    model.to(device)

    train_loss_ls = []
    val_loss_ls = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, lengths, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x, lengths)
            outputs = outputs.squeeze()

            loss = criterion(outputs, batch_y)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_loss_ls.append(train_loss)

        if val_loader is not None:
            _, _, _, mape, rmse = evaluate_sequences(model, val_loader, device=device)
            val_loss_ls.append(float(np.mean(np.array(rmse))))
        else:
            mape = None
            rmse = None

        if verbose:
            print(f"Epoch: {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}", end="")
            if mape is not None:
                print(
                    f", Validation: MAPE: {np.mean(np.array(mape)):.4f}, "
                    f"RMSE: {np.mean(np.array(rmse)):.4f}"
                )
            else:
                print()

    if val_loader is not None:
        return model, train_loss_ls, val_loss_ls
    return model, train_loss_ls


@overload
def evaluate_sequences(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = "cpu",
    metrics_only: Literal[False] = False,
) -> Tuple[list, list, list, List[float], List[float]]: ...


@overload
def evaluate_sequences(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = "cpu",
    *,
    metrics_only: Literal[True],
) -> Tuple[float, float]: ...


def evaluate_sequences(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = "cpu",
    metrics_only: bool = False,
) -> Union[Tuple[float, float], Tuple[list, list, list, List[float], List[float]]]:
    """Evaluate a sequence model with per-batch MAPE and RMSE.

    Metrics are the torchmetrics ``MeanAbsolutePercentageError`` and
    ``MeanSquaredError(squared=False)`` between ``model(x, lengths).squeeze()``
    and the squeezed target batch, as in the schu25a original. Inputs,
    predictions, and targets are additionally returned as nested Python lists
    for plotting.

    Args:
        model (nn.Module): Model called as ``model(x, lengths)``.
        val_loader (DataLoader): Loader yielding ``(x, lengths, y)`` batches.
            Batches are converted to NumPy, so ``device`` should be "cpu"
            unless ``metrics_only=True``.
        device (str, optional): Evaluation device. Defaults to "cpu".
        metrics_only (bool, optional): Return only ``(mean_mape, mean_rmse)``.
            Defaults to False.

    Returns:
        tuple: ``(x_ls, y_hat_ls, y_ls, mape_loss_ls, rmse_loss_ls)`` with one
            entry per batch, or ``(mean_mape, mean_rmse)`` if ``metrics_only``.

    Examples:
        ```{python}
        import pandas as pd
        import torch
        from torch.utils.data import DataLoader
        from spotoptim.data.manydataset import ManyToManyDataset, PadSequenceManyToMany
        from spotoptim.nn.many_to_many_rnn import ManyToManyRNN
        from spotoptim.nn.training import evaluate_sequences
        from spotoptim.utils.seed import seed_everything

        seed_everything(42)
        frames = [pd.DataFrame({"x": [0.1, 0.2, 0.3], "y": [1.0, 2.0, 3.0]})]
        ds = ManyToManyDataset(frames, target="y")
        dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=PadSequenceManyToMany())
        model = ManyToManyRNN(input_size=1, rnn_units=8, fc_units=8)
        x, y_hat, y, mape, rmse = evaluate_sequences(model, dl)
        print(len(y_hat), len(mape), len(rmse))
        ```
    """
    mean_abs_percentage_error = MeanAbsolutePercentageError().to(device)
    rmse_loss = MeanSquaredError(squared=False).to(device)

    model.eval()
    model.to(device)

    x_ls = []
    y_hat_ls = []
    y_ls = []
    mape_loss_ls = []
    rmse_loss_ls = []

    with torch.no_grad():
        for batch_x, lengths, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_hat = model(batch_x, lengths)
            y_hat = y_hat.squeeze()

            mape = mean_abs_percentage_error(y_hat, batch_y.squeeze())
            rmse = rmse_loss(y_hat, batch_y.squeeze())

            mape_loss_ls.append(mape.item())
            rmse_loss_ls.append(rmse.item())

            x_ls.append(batch_x.squeeze().detach().numpy().tolist())
            y_hat_ls.append(y_hat.squeeze().detach().numpy().tolist())
            y_ls.append(batch_y.squeeze().detach().numpy().tolist())

    if metrics_only:
        return float(np.mean(mape_loss_ls)), float(np.mean(rmse_loss_ls))
    return x_ls, y_hat_ls, y_ls, mape_loss_ls, rmse_loss_ls


def train_maps(
    model: nn.Module,
    maps: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int = 10,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[nn.Module, List[float]]:
    """Train a map-level model on a list of whole maps.

    The map-level analogue of the full-batch protocol in `train_sequences`:
    every epoch computes the loss of each map (between
    ``model(x, lengths).squeeze(-1)`` and the padded targets, unmasked like
    the original study code), averages over the maps, and takes ONE optimizer
    step. Use with models that consume one map per forward pass, e.g.
    `spotoptim.nn.map_context_rnn.MapContextRNN`.

    Args:
        model (nn.Module): Model called as ``model(x, lengths)`` on one map.
        maps (List[Tuple]): Training maps as ``(x, lengths, y)`` triples,
            e.g. from `spotoptim.data.manydataset.load_map_data`.
        optimizer (torch.optim.Optimizer): Optimizer over ``model.parameters()``.
        criterion (nn.Module): Loss module, e.g. ``nn.MSELoss()``.
        epochs (int, optional): Number of epochs. Defaults to 10.
        device (str, optional): Training device. Defaults to "cpu".
        verbose (bool, optional): Print per-epoch losses. Defaults to True.

    Returns:
        tuple: ``(model, train_loss_ls)`` with one mean-map loss per epoch.

    Examples:
        ```{python}
        import pandas as pd
        import torch
        import torch.nn as nn
        from spotoptim.data.manydataset import load_map_data
        from spotoptim.nn.map_context_rnn import MapContextRNN
        from spotoptim.nn.training import train_maps
        from spotoptim.utils.seed import seed_everything

        seed_everything(42)
        df = pd.DataFrame({
            "line": [1, 1, 1, 2, 2],
            "x": [0.1, 0.2, 0.3, 0.4, 0.5],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        maps = [load_map_data(df, target="y", group_by="line", drop="line")]
        model = MapContextRNN(input_size=1, rnn_units=8, fc_units=8, context_units=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model, losses = train_maps(
            model, maps, optimizer, nn.MSELoss(), epochs=2, verbose=False
        )
        print(len(losses))
        ```
    """
    model.train()
    model.to(device)

    train_loss_ls = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        losses = []
        for x, lengths, y in maps:
            outputs = model(x.to(device), lengths).squeeze(-1)
            losses.append(criterion(outputs, y.to(device)))
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        train_loss_ls.append(loss.item())

        if verbose:
            print(f"Epoch: {epoch + 1}/{epochs}, Train Loss: {loss.item():.4f}")

    return model, train_loss_ls


@overload
def evaluate_map(
    model: nn.Module,
    map_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: str = "cpu",
    metrics_only: Literal[False] = False,
) -> Tuple[list, list, list, List[float], List[float]]: ...


@overload
def evaluate_map(
    model: nn.Module,
    map_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: str = "cpu",
    *,
    metrics_only: Literal[True],
) -> Tuple[float, float]: ...


def evaluate_map(
    model: nn.Module,
    map_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: str = "cpu",
    metrics_only: bool = False,
) -> Union[Tuple[float, float], Tuple[list, list, list, List[float], List[float]]]:
    """Evaluate a map-level model on one held-out map, per line.

    The whole map is passed through the model in a single forward pass (an
    across-line context model needs all lines at once); MAPE and RMSE are
    then computed PER LINE on the first ``length`` points only, with the same
    torchmetrics semantics as `evaluate_sequences` on ``batch_size=1``
    loaders — the per-line values are directly comparable.

    Args:
        model (nn.Module): Model called as ``model(x, lengths)`` on one map.
        map_data (Tuple): The held-out map as an ``(x, lengths, y)`` triple,
            e.g. from `spotoptim.data.manydataset.load_map_data`.
        device (str, optional): Evaluation device. Defaults to "cpu".
        metrics_only (bool, optional): Return only ``(mean_mape, mean_rmse)``.
            Defaults to False.

    Returns:
        tuple: ``(x_ls, y_hat_ls, y_ls, mape_loss_ls, rmse_loss_ls)`` with one
            entry per line, or ``(mean_mape, mean_rmse)`` if ``metrics_only``.

    Examples:
        ```{python}
        import pandas as pd
        import torch
        from spotoptim.data.manydataset import load_map_data
        from spotoptim.nn.map_context_rnn import MapContextRNN
        from spotoptim.nn.training import evaluate_map
        from spotoptim.utils.seed import seed_everything

        seed_everything(42)
        df = pd.DataFrame({
            "line": [1, 1, 1, 2, 2],
            "x": [0.1, 0.2, 0.3, 0.4, 0.5],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        map_data = load_map_data(df, target="y", group_by="line", drop="line")
        model = MapContextRNN(input_size=1, rnn_units=8, fc_units=8, context_units=4)
        x, y_hat, y, mape, rmse = evaluate_map(model, map_data)
        print(len(y_hat), len(mape), len(rmse))
        ```
    """
    mean_abs_percentage_error = MeanAbsolutePercentageError().to(device)
    rmse_loss = MeanSquaredError(squared=False).to(device)

    model.eval()
    model.to(device)

    x, lengths, y = map_data
    with torch.no_grad():
        y_hat = model(x.to(device), lengths).squeeze(-1)

    x_ls = []
    y_hat_ls = []
    y_ls = []
    mape_loss_ls = []
    rmse_loss_ls = []

    for i, length in enumerate(lengths.tolist()):
        line_y_hat = y_hat[i, :length]
        line_y = y[i, :length].to(device)

        mape_loss_ls.append(mean_abs_percentage_error(line_y_hat, line_y).item())
        rmse_loss_ls.append(rmse_loss(line_y_hat, line_y).item())

        x_ls.append(x[i, :length].detach().cpu().numpy().tolist())
        y_hat_ls.append(line_y_hat.detach().cpu().numpy().tolist())
        y_ls.append(line_y.detach().cpu().numpy().tolist())

    if metrics_only:
        return float(np.mean(mape_loss_ls)), float(np.mean(rmse_loss_ls))
    return x_ls, y_hat_ls, y_ls, mape_loss_ls, rmse_loss_ls
