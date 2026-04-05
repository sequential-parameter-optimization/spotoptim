# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Serialization utilities for SpotOptim (save/load experiments and results)."""

import os
from typing import Optional

import dill


def get_result_filename(prefix: str) -> str:
    """Generate result filename with '_res.pkl' suffix."""
    if prefix is None:
        return "result_res.pkl"
    return f"{prefix}_res.pkl"


def get_experiment_filename(prefix: str) -> str:
    """Generate experiment filename with '_exp.pkl' suffix."""
    if prefix is None:
        return "experiment_exp.pkl"
    return f"{prefix}_exp.pkl"


def get_pickle_safe_optimizer(
    optimizer, unpickleables: str = "file_io", verbosity: int = 0
):
    """Create a pickle-safe copy of the optimizer.

    Args:
        optimizer: SpotOptim instance.
        unpickleables (str): "file_io" or "all". Defaults to "file_io".
        verbosity (int): 0=silent, 1=basic, 2=detailed. Defaults to 0.

    Returns:
        A copy of the optimizer with unpickleable components removed.
    """
    if unpickleables == "file_io":
        unpickleable_attrs = ["tb_writer"]
    else:
        unpickleable_attrs = ["tb_writer", "surrogate", "lhs_sampler"]

    picklable_state = {}

    for key, value in optimizer.__dict__.items():
        if key not in unpickleable_attrs:
            try:
                dill.dumps(value, protocol=dill.HIGHEST_PROTOCOL)
                picklable_state[key] = value
                if verbosity > 1:
                    print(f"Attribute '{key}' is picklable and will be included.")
            except Exception as e:
                if verbosity > 0:
                    print(
                        f"Attribute '{key}' is not picklable and will be excluded: {e}"
                    )
                continue
        else:
            if verbosity > 1:
                print(f"Attribute '{key}' explicitly excluded from pickling.")

    picklable_instance = optimizer.__class__.__new__(optimizer.__class__)
    picklable_instance.__dict__.update(picklable_state)

    for attr in unpickleable_attrs:
        if not hasattr(picklable_instance, attr):
            setattr(picklable_instance, attr, None)

    return picklable_instance


def reinitialize_components(optimizer) -> None:
    """Reinitialize components that were excluded during pickling.

    Recreates the surrogate model and LHS sampler.

    Args:
        optimizer: SpotOptim instance.
    """
    from scipy.stats.qmc import LatinHypercube

    if not hasattr(optimizer, "lhs_sampler") or optimizer.lhs_sampler is None:
        optimizer.lhs_sampler = LatinHypercube(d=optimizer.n_dim, rng=optimizer.seed)

    if not hasattr(optimizer, "surrogate") or optimizer.surrogate is None:
        optimizer.init_surrogate()


def save_result(
    optimizer,
    filename: Optional[str] = None,
    prefix: str = "result",
    path: Optional[str] = None,
    overwrite: bool = True,
    verbosity: int = 0,
) -> None:
    """Save complete optimization results to a pickle file.

    Args:
        optimizer: SpotOptim instance.
        filename (str, optional): Filename. If None, generates from prefix.
        prefix (str): Prefix for auto-generated filename. Defaults to "result".
        path (str, optional): Directory path. Defaults to None.
        overwrite (bool): Overwrite existing file. Defaults to True.
        verbosity (int): Verbosity level. Defaults to 0.
    """
    if filename is None:
        filename = get_result_filename(prefix)

    save_experiment(
        optimizer,
        filename=filename,
        path=path,
        overwrite=overwrite,
        unpickleables="file_io",
        verbosity=verbosity,
    )

    if path is not None:
        full_path = os.path.join(path, filename)
    else:
        full_path = filename
    print(f"Result saved to {full_path}")


def load_result(filename: str):
    """Load complete optimization results from a pickle file.

    Args:
        filename (str): Path to the result pickle file.

    Returns:
        SpotOptim: Loaded optimizer instance with complete results.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Result file not found: {filename}")

    try:
        with open(filename, "rb") as handle:
            optimizer = dill.load(handle)
        print(f"Loaded result from {filename}")
        reinitialize_components(optimizer)
        return optimizer
    except Exception as e:
        print(f"Error loading result: {e}")
        raise


def save_experiment(
    optimizer,
    filename: Optional[str] = None,
    prefix: str = "experiment",
    path: Optional[str] = None,
    overwrite: bool = True,
    unpickleables: str = "all",
    verbosity: int = 0,
) -> None:
    """Save experiment configuration to a pickle file.

    Args:
        optimizer: SpotOptim instance.
        filename (str, optional): Filename. If None, generates from prefix.
        prefix (str): Prefix for auto-generated filename. Defaults to "experiment".
        path (str, optional): Directory path. Defaults to None.
        overwrite (bool): Overwrite existing file. Defaults to True.
        unpickleables (str): "all" or "file_io". Defaults to "all".
        verbosity (int): Verbosity level. Defaults to 0.
    """
    optimizer._close_and_del_tensorboard_writer()

    optimizer_copy = get_pickle_safe_optimizer(
        optimizer, unpickleables=unpickleables, verbosity=verbosity
    )

    if filename is None:
        filename = get_experiment_filename(prefix)

    if path is not None:
        if not os.path.exists(path):
            os.makedirs(path)
        filename = os.path.join(path, filename)

    if os.path.exists(filename) and not overwrite:
        raise FileExistsError(
            f"File {filename} already exists. Use overwrite=True to overwrite."
        )

    try:
        with open(filename, "wb") as handle:
            dill.dump(optimizer_copy, handle, protocol=dill.HIGHEST_PROTOCOL)
        print(f"Experiment saved to {filename}")
    except Exception as e:
        print(f"Error during pickling: {e}")
        raise


def load_experiment(filename: str):
    """Load experiment configuration from a pickle file.

    Args:
        filename (str): Path to the experiment pickle file.

    Returns:
        SpotOptim: Loaded optimizer instance.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Experiment file not found: {filename}")

    try:
        with open(filename, "rb") as handle:
            optimizer = dill.load(handle)
        print(f"Loaded experiment from {filename}")
        reinitialize_components(optimizer)
        return optimizer
    except Exception as e:
        print(f"Error loading experiment: {e}")
        raise
