# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

import numpy as np
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from spotoptim import SpotOptim


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def dummy_func(X):
    return np.sum(X**2, axis=1)


def _count_scalar_steps(logdir, tag="success_rate"):
    """Return the number of distinct steps logged for a scalar tag.

    add_scalar events land in the run root; add_hparams events land in
    per-call subdirectories, so every event file under ``logdir`` is read.
    """
    steps = set()
    for root, _dirs, files in os.walk(str(logdir)):
        if any(f.startswith("events.out.tfevents") for f in files):
            acc = EventAccumulator(root)
            acc.Reload()
            if tag in acc.Tags().get("scalars", []):
                steps.update(e.step for e in acc.Scalars(tag))
    return len(steps)


def test_tensorboard_enabled_in_parallel(capsys):
    """Test that TensorBoard is ENABLED when n_jobs > 1 (steady-state)."""
    opt = SpotOptim(
        fun=dummy_func,
        bounds=[(-5, 5)],
        n_initial=4,
        max_iter=6,
        n_jobs=2,
        tensorboard_log=True,
        verbose=True,
    )

    # Manually trigger initialization (usually called at start of optimize, after initial design)
    # We need to simulate having some data for it to log
    opt.X_ = np.array([[0.0], [1.0], [2.0], [3.0]])
    opt.y_ = np.array([0.0, 1.0, 4.0, 9.0])

    # Initialize stats that _write_tensorboard_scalars expects
    opt.min_y = 0.0
    opt.min_mean_y = 0.0
    opt.min_var_y = 0.0
    opt.min_mean_X = opt.X_[0]
    opt.min_X = opt.X_[0]
    opt.success_rate = 0.0
    opt._init_tensorboard()

    # Check that tb_writer is NOT None
    assert opt.tb_writer is not None, "tb_writer should be enabled when n_jobs > 1"

    # Check that config was updated
    assert opt.config.tensorboard_log is True, "config.tensorboard_log should stay True"

    # Check for enabled message
    captured = capsys.readouterr()
    assert "TensorBoard logging enabled" in captured.out

    # Ensure optimization runs without pickling error
    # (SpotOptim handles tb_writer removal during dill serialization)
    opt.optimize()


def test_parallel_logs_infill_evals(tmp_path):
    """Steady-state parallel runs log scalars beyond the initial design.

    Regression test for the gap where workers carry ``tb_writer=None`` and
    the parent result loop never wrote per-eval scalars, so parallel runs
    logged only the initial design (one step).
    """
    n_initial, max_iter = 4, 12
    path = str(tmp_path / "tb_parallel")
    opt = SpotOptim(
        fun=dummy_func,
        bounds=[(-5, 5)],
        n_initial=n_initial,
        max_iter=max_iter,
        n_jobs=2,
        tensorboard_log=True,
        tensorboard_path=path,
        seed=0,
        verbose=False,
    )
    opt.optimize()

    assert len(opt.y_) > n_initial, "expected infill evaluations beyond n_initial"
    # Before the fix, success_rate appeared at a single step (initial design
    # only). With per-eval parent-side logging it advances with each batch.
    assert _count_scalar_steps(path, "success_rate") > 1


def test_parallel_no_tensorboard_regression(tmp_path):
    """tensorboard_log=False parallel run is unaffected: no writer, no runs dir."""
    opt = SpotOptim(
        fun=dummy_func,
        bounds=[(-5, 5)],
        n_initial=4,
        max_iter=10,
        n_jobs=2,
        tensorboard_log=False,
        seed=0,
        verbose=False,
    )
    res = opt.optimize()

    assert opt.tb_writer is None
    assert res.success is True
    assert not (tmp_path / "runs").exists()


def test_tensorboard_enabled_in_sequential():
    """Test that TensorBoard IS enabled when n_jobs = 1."""
    opt = SpotOptim(
        fun=dummy_func,
        bounds=[(-5, 5)],
        n_initial=4,
        max_iter=6,
        n_jobs=1,
        tensorboard_log=True,
        verbose=False,
    )

    assert opt.tb_writer is not None, "tb_writer should NOT be None when n_jobs = 1"
    # Cleanup
    if opt.tb_writer:
        opt.tb_writer.close()
