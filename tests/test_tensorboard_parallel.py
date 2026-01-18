import numpy as np
import pytest
from spotoptim import SpotOptim

def dummy_func(X):
    return np.sum(X**2, axis=1)

def test_tensorboard_disabled_in_parallel(capsys):
    """Test that TensorBoard is disabled when n_jobs > 1."""
    opt = SpotOptim(
        fun=dummy_func,
        bounds=[(-5, 5)],
        n_initial=4,
        max_iter=6,
        n_jobs=2,
        tensorboard_log=True,
        verbose=True
    )
    
    # Check that tb_writer is None
    assert opt.tb_writer is None, "tb_writer should be None when n_jobs > 1"
    
    # Check that config was updated (though property access redirects to config)
    # The fix updates config.tensorboard_log
    assert opt.config.tensorboard_log is False, "config.tensorboard_log should be set to False"
    
    # Check for warning message
    captured = capsys.readouterr()
    assert "Warning: TensorBoard logging disabled" in captured.out
    
    # Ensure optimization runs without pickling error
    # This would crash if tb_writer was present
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
