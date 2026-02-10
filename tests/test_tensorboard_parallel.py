# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pytest
from spotoptim import SpotOptim

def dummy_func(X):
    return np.sum(X**2, axis=1)

def test_tensorboard_enabled_in_parallel(capsys):
    """Test that TensorBoard is ENABLED when n_jobs > 1 (steady-state)."""
    opt = SpotOptim(
        fun=dummy_func,
        bounds=[(-5, 5)],
        n_initial=4,
        max_iter=6,
        n_jobs=2,
        tensorboard_log=True,
        verbose=True
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


def test_tensorboard_enabled_in_sequential():
    """Test that TensorBoard IS enabled when n_jobs = 1."""
    opt = SpotOptim(
        fun=dummy_func,
        bounds=[(-5, 5)],
        n_initial=4,
        max_iter=6,
        n_jobs=1,
        tensorboard_log=True,
        verbose=False
    )
    
    assert opt.tb_writer is not None, "tb_writer should NOT be None when n_jobs = 1"
    # Cleanup
    if opt.tb_writer:
        opt.tb_writer.close()
