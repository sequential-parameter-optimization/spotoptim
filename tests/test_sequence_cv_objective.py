# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotoptim.function.sequence_cv_objective."""

import numpy as np
import pandas as pd
import pytest

from spotoptim.core.data import SpotDataFromTorchDataset
from spotoptim.core.experiment import ExperimentControl
from spotoptim.data.manydataset import ManyToManyDataset
from spotoptim.function.sequence_cv_objective import SequenceCVObjective
from spotoptim.hyperparameters import ParameterSet
from spotoptim.nn.many_to_many_rnn import ManyToManyRNNRegressor


def _experiment(seed=42):
    rng = np.random.default_rng(0)
    frames = [pd.DataFrame({"x": rng.random(n), "y": rng.random(n)}) for n in (4, 3, 5)]
    ds = ManyToManyDataset(frames, target="y")
    params = ParameterSet()
    params.add_int("rnn_units", 1, 3)
    params.add_int("epochs", 1, 2)
    params.add_int("batch_size", 1, 2)
    params.add_factor("optimizer", ["Adam", "SGD"], default="Adam")
    params.add_float("lr_mult", 0.1, 2.0)
    return ExperimentControl(
        dataset=SpotDataFromTorchDataset(ds, input_dim=1, output_dim=1),
        model_class=ManyToManyRNNRegressor,
        hyperparameters=params,
        metrics=["val_loss"],
        seed=seed,
    )


def test_objective_returns_finite_mean_loocv_loss():
    objective = SequenceCVObjective(_experiment())
    # rnn_units=2, epochs=2, batch_size=2, optimizer="Adam", lr_mult=1.0
    y = objective(np.array([[2, 2, 2, "Adam", 1.0]], dtype=object))
    assert y.shape == (1, 1)
    assert np.isfinite(y[0, 0])


def test_objective_is_reproducible_with_fixed_seed():
    y1 = SequenceCVObjective(_experiment())(
        np.array([[2, 2, 2, "Adam", 1.0]], dtype=object)
    )
    y2 = SequenceCVObjective(_experiment())(
        np.array([[2, 2, 2, "Adam", 1.0]], dtype=object)
    )
    assert y1[0, 0] == pytest.approx(y2[0, 0])


def test_param_mappers_apply_power_of_two():
    captured = {}

    class SpyRegressor(ManyToManyRNNRegressor):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(**kwargs)

    exp = _experiment()
    exp.model_class = SpyRegressor
    objective = SequenceCVObjective(
        exp, param_mappers={"rnn_units": lambda v: 2 ** int(v)}
    )
    y = objective(np.array([[3, 1, 2, "SGD", 1.0]], dtype=object))
    assert np.isfinite(y[0, 0])
    assert captured["rnn_units"] == 8


def test_failing_configuration_evaluates_to_nan():
    objective = SequenceCVObjective(
        _experiment(), param_mappers={"optimizer": lambda v: "Bogus"}
    )
    y = objective(np.array([[2, 1, 1, "Adam", 1.0]], dtype=object))
    assert y.shape == (1, 1)
    assert np.isnan(y[0, 0])


def test_factor_index_decoding_without_strings():
    # SpotOptim can also pass numeric factor indices; index 1 -> "SGD"
    objective = SequenceCVObjective(_experiment())
    y = objective(np.array([[2, 1, 1, 1, 1.0]], dtype=float))
    assert np.isfinite(y[0, 0])
