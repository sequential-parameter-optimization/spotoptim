# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""TensorBoard logging utilities for SpotOptim.

All functions receive an optimizer object and read/write its attributes.
This module is used exclusively by SpotOptim's private TensorBoard methods.
"""

import os
import shutil

import numpy as np


def clean_tensorboard_logs(optimizer) -> None:
    """Clean old TensorBoard log directories from the runs folder.

    Removes all subdirectories in the 'runs' directory if tensorboard_clean is True.

    Args:
        optimizer: SpotOptim instance.
    """
    if optimizer.tensorboard_clean:
        runs_dir = "runs"
        if os.path.exists(runs_dir) and os.path.isdir(runs_dir):
            subdirs = [
                os.path.join(runs_dir, d)
                for d in os.listdir(runs_dir)
                if os.path.isdir(os.path.join(runs_dir, d))
            ]

            if subdirs:
                removed_count = 0
                for subdir in subdirs:
                    try:
                        shutil.rmtree(subdir)
                        removed_count += 1
                        if optimizer.verbose:
                            print(f"Removed old TensorBoard logs: {subdir}")
                    except Exception as e:
                        if optimizer.verbose:
                            print(f"Warning: Could not remove {subdir}: {e}")

                if optimizer.verbose and removed_count > 0:
                    print(
                        f"Cleaned {removed_count} old TensorBoard log director{'y' if removed_count == 1 else 'ies'}"
                    )
            elif optimizer.verbose:
                print("No old TensorBoard logs to clean in 'runs' directory")
        elif optimizer.verbose:
            print("'runs' directory does not exist, nothing to clean")


def init_tensorboard_writer(optimizer) -> None:
    """Initialize TensorBoard SummaryWriter if logging is enabled.

    Creates a unique log directory based on timestamp if tensorboard_log is True.

    Args:
        optimizer: SpotOptim instance.
    """
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime

    if optimizer.tensorboard_log:
        if optimizer.tensorboard_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            optimizer.tensorboard_path = f"runs/spotoptim_{timestamp}"

        os.makedirs(optimizer.tensorboard_path, exist_ok=True)

        optimizer.tb_writer = SummaryWriter(log_dir=optimizer.tensorboard_path)
        if optimizer.verbose:
            print(f"TensorBoard logging enabled: {optimizer.tensorboard_path}")
    else:
        optimizer.tb_writer = None
        if optimizer.verbose:
            print("TensorBoard logging disabled")


def write_tensorboard_scalars(optimizer) -> None:
    """Write scalar metrics to TensorBoard.

    Logs min_y, last y, best X coordinates, success rate.
    For noisy optimization, also logs mean values and variance.

    Args:
        optimizer: SpotOptim instance.
    """
    if optimizer.tb_writer is None or optimizer.y_ is None or len(optimizer.y_) == 0:
        return

    step = optimizer.counter
    y_last = optimizer.y_[-1]

    if not (optimizer.repeats_initial > 1) or (optimizer.repeats_surrogate > 1):
        optimizer.tb_writer.add_scalars(
            "y_values", {"min": optimizer.min_y, "last": y_last}, step
        )
        optimizer.tb_writer.add_scalar("success_rate", optimizer.success_rate, step)
        for i in range(optimizer.n_dim):
            param_name = optimizer.var_name[i] if optimizer.var_name else f"x{i}"
            optimizer.tb_writer.add_scalar(
                f"X_best/{param_name}", optimizer.min_X[i], step
            )
    else:
        optimizer.tb_writer.add_scalars(
            "y_values",
            {"min": optimizer.min_y, "mean_best": optimizer.min_mean_y, "last": y_last},
            step,
        )
        optimizer.tb_writer.add_scalar("y_variance_at_best", optimizer.min_var_y, step)
        optimizer.tb_writer.add_scalar("success_rate", optimizer.success_rate, step)

        for i in range(optimizer.n_dim):
            param_name = optimizer.var_name[i] if optimizer.var_name else f"x{i}"
            optimizer.tb_writer.add_scalar(
                f"X_mean_best/{param_name}", optimizer.min_mean_X[i], step
            )

    optimizer.tb_writer.flush()


def write_tensorboard_hparams(optimizer, X: np.ndarray, y: float) -> None:
    """Write hyperparameters and metric to TensorBoard.

    Args:
        optimizer: SpotOptim instance.
        X (ndarray): Design point coordinates, shape (n_features,).
        y (float): Function value at X.
    """
    if optimizer.tb_writer is None:
        return

    hparam_dict = {optimizer.var_name[i]: float(X[i]) for i in range(optimizer.n_dim)}
    metric_dict = {"hp_metric": float(y)}

    optimizer.tb_writer.add_hparams(hparam_dict, metric_dict)
    optimizer.tb_writer.flush()


def close_tensorboard_writer(optimizer) -> None:
    """Close TensorBoard writer and cleanup.

    Args:
        optimizer: SpotOptim instance.
    """
    if hasattr(optimizer, "tb_writer") and optimizer.tb_writer is not None:
        optimizer.tb_writer.flush()
        optimizer.tb_writer.close()
        if optimizer.verbose:
            print(
                f"TensorBoard writer closed. View logs with: tensorboard --logdir={optimizer.tensorboard_path}"
            )
        del optimizer.tb_writer


def init_tensorboard(optimizer) -> None:
    """Log initial design to TensorBoard.

    Logs all initial design points (hyperparameters and function values)
    and scalar metrics. Only executes if TensorBoard logging is enabled.

    Args:
        optimizer: SpotOptim instance.
    """
    if optimizer.tensorboard_log and optimizer.tb_writer is None:
        if optimizer.tensorboard_path:
            log_dir = optimizer.tensorboard_path
        else:
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"runs/spotoptim_{timestamp}"
            optimizer.config.tensorboard_path = log_dir
            optimizer.tensorboard_path = log_dir

        try:
            from torch.utils.tensorboard import SummaryWriter

            optimizer.tb_writer = SummaryWriter(log_dir=log_dir)
            if optimizer.verbose:
                print(f"TensorBoard logging enabled: {log_dir}")
        except ImportError:
            print("Warning: torch or tensorboard not installed. Logging disabled.")
            optimizer.tb_writer = None
            optimizer.config.tensorboard_log = False
            optimizer.tensorboard_log = False

    if optimizer.tb_writer is not None:
        for i in range(len(optimizer.y_)):
            write_tensorboard_hparams(optimizer, optimizer.X_[i], optimizer.y_[i])
        write_tensorboard_scalars(optimizer)


def close_and_del_tensorboard_writer(optimizer) -> None:
    """Close and delete TensorBoard writer to prepare for pickling.

    Args:
        optimizer: SpotOptim instance.
    """
    if hasattr(optimizer, "tb_writer") and optimizer.tb_writer is not None:
        try:
            optimizer.tb_writer.flush()
            optimizer.tb_writer.close()
        except Exception:
            pass
        optimizer.tb_writer = None
