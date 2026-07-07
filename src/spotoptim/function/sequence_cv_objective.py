# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Leave-one-sequence-out cross-validation objective for sequence regression.

Ported from the training path used by ``spotpython.fun.hyperlight.HyperLight``
with ``fun_control["hacky"]=True`` (``spotpython.light.trainmodel``), without
Lightning: for every sequence of a `spotoptim.data.manydataset.ManyToManyDataset`,
a fresh model is trained on all remaining sequences for the full number of
epochs and evaluated on the held-out sequence; the objective value is the mean
of the held-out losses. As in spotPython, training runs the complete epoch
budget (the ``patience`` hyperparameter is accepted but has no effect, because
the reference implementation never attached a validation loader during
fitting) and the loss is computed on zero-padded batches without masking.

Requires the ``torch`` optional extra (``pip install 'spotoptim[torch]'``).
"""

import logging
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset

from spotoptim.core.experiment import ExperimentControl
from spotoptim.function.torch_objective import TorchObjective
from spotoptim.nn.optimizer import optimizer_handler

logger = logging.getLogger(__name__)


class SequenceCVObjective(TorchObjective):
    """Callable SpotOptim objective running leave-one-sequence-out CV.

    The experiment's dataset must be a
    `spotoptim.core.data.SpotDataFromTorchDataset` whose training dataset
    yields variable-length ``(features, targets)`` sequence pairs (e.g. a
    `spotoptim.data.manydataset.ManyToManyDataset`). For each hyperparameter
    configuration, every sequence is held out once: a fresh model is trained
    on the remaining sequences and scored on the held-out one, and the mean
    held-out loss is returned. A configuration whose training fails (e.g. an
    optimizer incompatible with dense gradients) evaluates to ``np.nan``, to
    be repaired by SpotOptim's ``penalty`` handling.

    The tuned hyperparameters follow the spotPython ``ManyToManyRNNRegressor``
    convention: ``epochs``, ``batch_size``, ``optimizer``, and ``lr_mult``
    drive training via `spotoptim.nn.optimizer.optimizer_handler` and a
    ``MultiStepLR`` schedule (three milestones at 1/4, 2/4, and 3/4 of the
    epoch budget, decay factor 0.1); all remaining parameters are passed to
    the model constructor.

    Args:
        experiment (ExperimentControl): Experiment configuration; its
            ``dataset`` wraps the sequence dataset and its ``model_class`` is
            instantiated once per fold.
        seed (Optional[int]): Random seed applied before each configuration
            evaluation. Falls back to ``experiment.seed`` when None.
            Defaults to None.
        collate_fn (Optional[Callable]): Batch collate producing
            ``(padded_x, lengths, padded_y)``. Defaults to
            `spotoptim.data.manydataset.PadSequenceManyToMany`.
        param_mappers (Optional[Dict[str, Callable]]): Per-parameter functions
            applied to the decoded hyperparameter values before training,
            e.g. ``{"epochs": lambda v: 2 ** int(v)}`` to reproduce
            spotPython's ``transform_power_2_int``. Defaults to None.

    Examples:
        ```{python}
        import numpy as np
        import pandas as pd
        from spotoptim.core.data import SpotDataFromTorchDataset
        from spotoptim.core.experiment import ExperimentControl
        from spotoptim.data.manydataset import ManyToManyDataset
        from spotoptim.function.sequence_cv_objective import SequenceCVObjective
        from spotoptim.hyperparameters import ParameterSet
        from spotoptim.nn.many_to_many_rnn import ManyToManyRNNRegressor

        rng = np.random.default_rng(0)
        frames = [
            pd.DataFrame({"x": rng.random(4), "y": rng.random(4)})
            for _ in range(3)
        ]
        ds = ManyToManyDataset(frames, target="y")
        params = ParameterSet()
        params.add_int("epochs", 1, 2)
        params.add_float("lr_mult", 0.1, 1.0)
        exp = ExperimentControl(
            dataset=SpotDataFromTorchDataset(ds, input_dim=1, output_dim=1),
            model_class=ManyToManyRNNRegressor,
            hyperparameters=params,
            seed=42,
        )
        objective = SequenceCVObjective(exp)
        y = objective(np.array([[2, 1.0]]))
        print(y.shape)
        ```
    """

    def __init__(
        self,
        experiment: ExperimentControl,
        seed: Optional[int] = None,
        collate_fn: Optional[Callable] = None,
        param_mappers: Optional[Dict[str, Callable]] = None,
    ):
        super().__init__(experiment=experiment, seed=seed)
        if collate_fn is None:
            from spotoptim.data.manydataset import PadSequenceManyToMany

            collate_fn = PadSequenceManyToMany()
        self.collate_fn = collate_fn
        self.param_mappers = param_mappers or {}

    def decode_params(self, X_row: np.ndarray) -> Dict[str, Any]:
        """Decode one parameter vector and apply the configured mappers.

        Args:
            X_row (np.ndarray): One row of the optimizer's input array, in
                natural scale with factor levels as strings.

        Returns:
            Dict[str, Any]: Hyperparameter dictionary ready for training.
        """
        params = self._get_hyperparameters(X_row)
        for name, mapper in self.param_mappers.items():
            if name in params:
                params[name] = mapper(params[name])
        return params

    def _build_model(self, params: Dict[str, Any]) -> nn.Module:
        """Instantiate a fresh model for one fold.

        Args:
            params (Dict[str, Any]): Decoded hyperparameters.

        Returns:
            nn.Module: The model instance.
        """
        dataset = self.experiment.dataset
        model_kwargs = {
            "input_dim": dataset.input_dim,
            "output_dim": dataset.output_dim,
        }
        model_kwargs.update(params)
        return self.experiment.model_class(**model_kwargs)

    def _train_fold(
        self, model: nn.Module, train_loader: DataLoader, params: Dict[str, Any]
    ) -> None:
        """Train one fold for the full epoch budget.

        Replicates the spotPython reference: one optimizer step per batch, one
        ``MultiStepLR`` scheduler step per epoch, and the loss computed on the
        zero-padded targets reshaped to the prediction shape.

        Args:
            model (nn.Module): The model to train (modified in place).
            train_loader (DataLoader): Loader over the training sequences.
            params (Dict[str, Any]): Decoded hyperparameters; consumes
                ``epochs``, ``optimizer``, and ``lr_mult``.
        """
        epochs = int(params.get("epochs", self.experiment.epochs or 100))
        optimizer = optimizer_handler(
            optimizer_name=str(params.get("optimizer", "SGD")),
            params=model.parameters(),
            lr_mult=float(params.get("lr_mult", 1.0)),
        )
        num_milestones = 3
        milestones = [
            int(epochs / (num_milestones + 1) * (i + 1)) for i in range(num_milestones)
        ]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        criterion = self.experiment.loss_function or nn.MSELoss()

        model.to(self.device)
        model.train()
        for _ in range(epochs):
            for x, lengths, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                y_hat = model(x, lengths)
                loss = criterion(y_hat, y.view_as(y_hat))
                loss.backward()
                optimizer.step()
            scheduler.step()

    def _validate_fold(self, model: nn.Module, val_loader: DataLoader) -> float:
        """Compute the mean loss of one fold's held-out data.

        Args:
            model (nn.Module): The trained model.
            val_loader (DataLoader): Loader over the held-out sequence(s).

        Returns:
            float: Mean loss over the loader's batches.
        """
        criterion = self.experiment.loss_function or nn.MSELoss()
        model.eval()
        losses = []
        with torch.no_grad():
            for x, lengths, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_hat = model(x, lengths)
                losses.append(criterion(y_hat, y.view_as(y_hat)).item())
        return float(np.mean(losses))

    def _evaluate_config(self, params: Dict[str, Any]) -> float:
        """Run the full leave-one-sequence-out CV for one configuration.

        Args:
            params (Dict[str, Any]): Decoded hyperparameters.

        Returns:
            float: Mean held-out loss across all folds.
        """
        dataset = self.experiment.dataset.get_train_data()
        batch_size = int(params.get("batch_size", self.experiment.batch_size))
        indices = list(range(len(dataset)))
        fold_losses = []
        for i in indices:
            train_indices = [j for j in indices if j != i]
            train_loader = DataLoader(
                Subset(dataset, train_indices),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
                num_workers=self.experiment.num_workers,
            )
            val_loader = DataLoader(
                Subset(dataset, [i]),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
                num_workers=self.experiment.num_workers,
            )
            model = self._build_model(params)
            self._train_fold(model, train_loader, params)
            fold_loss = self._validate_fold(model, val_loader)
            fold_losses.append(fold_loss)
            if self.experiment.verbosity > 0:
                print(f"SequenceCVObjective: fold {i} val_loss: {fold_loss}")
        return float(np.mean(fold_losses))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the objective for an array of configurations.

        Args:
            X (np.ndarray): Input array of shape ``(n_samples, n_params)`` or
                ``(n_params,)``, in natural scale with factor levels as strings.

        Returns:
            np.ndarray: Array of shape ``(n_samples, 1)`` with the mean
                held-out loss per configuration; ``np.nan`` where the
                evaluation failed.
        """
        X = np.atleast_2d(X)
        results = []
        for i in range(X.shape[0]):
            if self.seed is not None:
                self._set_seed(self.seed)
            params = self.decode_params(X[i])
            if self.experiment.verbosity > 0:
                print(f"SequenceCVObjective: config: {params}")
            try:
                value = self._evaluate_config(params)
            except Exception as err:
                logger.error(
                    "SequenceCVObjective: evaluation failed (%s: %s); returning nan",
                    type(err).__name__,
                    err,
                )
                value = float("nan")
            results.append([value])
        return np.array(results)
