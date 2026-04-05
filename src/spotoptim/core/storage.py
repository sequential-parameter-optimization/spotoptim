# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Data management utilities for optimization storage, stats, and aggregation."""

from typing import Tuple

import numpy as np


def init_storage(optimizer, X0: np.ndarray, y0: np.ndarray) -> None:
    """Initialize storage for optimization.

    Sets up the initial data structures needed for optimization tracking.

    Args:
        optimizer: SpotOptim instance.
        X0 (ndarray): Initial design points in internal scale, shape (n_samples, n_features).
        y0 (ndarray): Function values at X0, shape (n_samples,).
    """
    optimizer.X_ = optimizer.inverse_transform_X(X0.copy())
    optimizer.y_ = y0.copy()
    optimizer.n_iter_ = 0


def update_storage(optimizer, X_new: np.ndarray, y_new: np.ndarray) -> None:
    """Update storage (X_, y_) with new evaluation points.

    Appends new design points and their function values to the storage arrays.
    Points are converted from internal scale to original scale before storage.

    Args:
        optimizer: SpotOptim instance.
        X_new (ndarray): New design points in internal scale, shape (n_new, n_features).
        y_new (ndarray): Function values at X_new, shape (n_new,).
    """
    optimizer.X_ = np.vstack([optimizer.X_, optimizer.inverse_transform_X(X_new)])
    optimizer.y_ = np.append(optimizer.y_, y_new)


def update_stats(optimizer) -> None:
    """Update optimization statistics.

    Updates min_y, min_X, counter, and aggregated stats for noisy functions.

    Args:
        optimizer: SpotOptim instance.
    """
    if optimizer.y_ is None or len(optimizer.y_) == 0:
        return

    # Basic stats
    optimizer.min_y = np.min(optimizer.y_)
    optimizer.min_X = optimizer.X_[np.argmin(optimizer.y_)]
    optimizer.counter = len(optimizer.y_)

    # Aggregated stats for noisy functions
    if (optimizer.repeats_initial > 1) or (optimizer.repeats_surrogate > 1):
        optimizer.mean_X, optimizer.mean_y, optimizer.var_y = aggregate_mean_var(
            optimizer, optimizer.X_, optimizer.y_
        )
        # X value of the best mean y value so far
        best_mean_idx = np.argmin(optimizer.mean_y)
        optimizer.min_mean_X = optimizer.mean_X[best_mean_idx]
        # Best mean y value so far
        optimizer.min_mean_y = optimizer.mean_y[best_mean_idx]
        # Variance of the best mean y value so far
        optimizer.min_var_y = optimizer.var_y[best_mean_idx]


def update_success_rate(optimizer, y_new: np.ndarray) -> None:
    """Update the rolling success rate of the optimization process.

    A success is counted only if the new value is better (smaller) than the best
    found y value so far. Should be called BEFORE updating optimizer.y_.

    Args:
        optimizer: SpotOptim instance.
        y_new (ndarray): The new function values to consider.
    """
    if not hasattr(optimizer, "_success_history") or optimizer._success_history is None:
        optimizer._success_history = []

    if optimizer.y_ is not None and len(optimizer.y_) > 0:
        best_y_before = min(optimizer.y_)
    else:
        best_y_before = float("inf")

    successes = []
    current_best = best_y_before

    for val in y_new:
        if val < current_best:
            successes.append(1)
            current_best = val
        else:
            successes.append(0)

    optimizer._success_history.extend(successes)
    optimizer._success_history = optimizer._success_history[-optimizer.window_size :]

    window_size = len(optimizer._success_history)
    num_successes = sum(optimizer._success_history)
    optimizer.success_rate = num_successes / window_size if window_size > 0 else 0.0


def get_success_rate(optimizer) -> float:
    """Get the current success rate of the optimization process.

    Args:
        optimizer: SpotOptim instance.

    Returns:
        float: The current success rate.
    """
    return float(getattr(optimizer, "success_rate", 0.0) or 0.0)


def aggregate_mean_var(
    optimizer, X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate X and y values to compute mean and variance per group.

    For repeated evaluations at the same design point, computes the mean
    function value and variance (using population variance, ddof=0).

    Args:
        optimizer: SpotOptim instance (unused, kept for API consistency).
        X (ndarray): Design points, shape (n_samples, n_features).
        y (ndarray): Function values, shape (n_samples,).

    Returns:
        tuple: (X_agg, y_mean, y_var).
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("Invalid input shapes for aggregate_mean_var")

    if X.shape[0] == 0:
        return np.empty((0, X.shape[1])), np.array([]), np.array([])

    _, unique_idx, inverse_idx = np.unique(
        X, axis=0, return_index=True, return_inverse=True
    )

    X_agg = X[unique_idx]

    n_groups = len(unique_idx)
    y_mean = np.zeros(n_groups)
    y_var = np.zeros(n_groups)

    for i in range(n_groups):
        group_mask = inverse_idx == i
        group_y = y[group_mask]
        y_mean[i] = np.mean(group_y)
        y_var[i] = np.var(group_y, ddof=0)

    return X_agg, y_mean, y_var
