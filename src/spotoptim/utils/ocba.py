# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Optimal Computing Budget Allocation (OCBA) utilities for noisy optimization.

References:
    Chun-Hung Chen and Loo Hay Lee: Stochastic Simulation Optimization:
    An Optimal Computer Budget Allocation, pp. 49 and pp. 215
"""

from typing import Optional

import numpy as np


def get_ranks(x: np.ndarray) -> np.ndarray:
    """Returns ranks of numbers within input array x.

    Args:
        x (ndarray): Input array.

    Returns:
        ndarray: Ranks array where ranks[i] is the rank of x[i].
    """
    ts = x.argsort()
    ranks = np.empty_like(ts)
    ranks[ts] = np.arange(len(x))
    return ranks


def get_ocba(
    means: np.ndarray, vars: np.ndarray, delta: int, verbose: bool = False
) -> Optional[np.ndarray]:
    """Optimal Computing Budget Allocation (OCBA).

    Calculates budget recommendations for given means, variances, and incremental
    budget using the OCBA algorithm.

    Args:
        means (ndarray): Array of means.
        vars (ndarray): Array of variances.
        delta (int): Incremental budget.
        verbose (bool): If True, print debug information. Defaults to False.

    Returns:
        Optional[ndarray]: Array of budget recommendations, or None if conditions not met.
    """
    if np.all(vars > 0) and (means.shape[0] > 2):
        n_designs = means.shape[0]
        allocations = np.zeros(n_designs, np.int32)
        ratios = np.zeros(n_designs, np.float64)
        budget = delta
        ranks = get_ranks(means)
        best, second_best = np.argpartition(ranks, 2)[:2]
        ratios[second_best] = 1.0
        select = [i for i in range(n_designs) if i not in [best, second_best]]
        temp = (means[best] - means[second_best]) / (means[best] - means[select])
        ratios[select] = np.square(temp) * (vars[select] / vars[second_best])
        select = [i for i in range(n_designs) if i not in [best]]
        temp = (np.square(ratios[select]) / vars[select]).sum()
        ratios[best] = np.sqrt(vars[best] * temp)
        more_runs = np.full(n_designs, True, dtype=bool)
        add_budget = np.zeros(n_designs, dtype=float)
        more_alloc = True

        if verbose:
            print("\nIn get_ocba():")
            print(f"means: {means}")
            print(f"vars: {vars}")
            print(f"delta: {delta}")
            print(f"n_designs: {n_designs}")
            print(f"Ratios: {ratios}")
            print(f"Best: {best}, Second best: {second_best}")

        while more_alloc:
            more_alloc = False
            ratio_s = (more_runs * ratios).sum()
            add_budget[more_runs] = (budget / ratio_s) * ratios[more_runs]
            add_budget = np.around(add_budget).astype(int)
            mask = add_budget < allocations
            add_budget[mask] = allocations[mask]
            more_runs[mask] = 0

            if mask.sum() > 0:
                more_alloc = True
            if more_alloc:
                budget = allocations.sum() + delta
                budget -= (add_budget * ~more_runs).sum()

        t_budget = add_budget.sum()

        # Adjust the best design to match the exact delta
        adjustment = allocations.sum() + delta - t_budget
        add_budget[best] = max(allocations[best], add_budget[best] + adjustment)

        return add_budget - allocations
    else:
        return None


def get_ocba_X(
    X: np.ndarray,
    means: np.ndarray,
    vars: np.ndarray,
    delta: int,
    verbose: bool = False,
) -> Optional[np.ndarray]:
    """Calculate OCBA allocation and repeat input array X accordingly.

    Args:
        X (ndarray): Input array to be repeated, shape (n_designs, n_features).
        means (ndarray): Array of means for each design.
        vars (ndarray): Array of variances for each design.
        delta (int): Incremental budget.
        verbose (bool): If True, print debug information. Defaults to False.

    Returns:
        Optional[ndarray]: Repeated array of X based on OCBA allocation, or None.
    """
    if np.all(vars > 0) and (means.shape[0] > 2):
        o = get_ocba(means=means, vars=vars, delta=delta, verbose=verbose)
        return np.repeat(X, o, axis=0)
    else:
        return None


def apply_ocba(optimizer) -> Optional[np.ndarray]:
    """Apply Optimal Computing Budget Allocation for noisy functions.

    Determines which existing design points should be re-evaluated based on
    the OCBA algorithm.

    Args:
        optimizer: SpotOptim instance.

    Returns:
        Optional[ndarray]: Array of design points to re-evaluate, or None.
    """
    X_ocba = None
    if (
        optimizer.repeats_initial > 1 or optimizer.repeats_surrogate > 1
    ) and optimizer.ocba_delta > 0:
        if not np.all(optimizer.var_y > 0) and (optimizer.mean_X.shape[0] <= 2):
            if optimizer.verbose:
                print("Warning: OCBA skipped (need >2 points with variance > 0)")
        elif np.all(optimizer.var_y > 0) and (optimizer.mean_X.shape[0] > 2):
            X_ocba = get_ocba_X(
                optimizer.mean_X,
                optimizer.mean_y,
                optimizer.var_y,
                optimizer.ocba_delta,
                verbose=optimizer.verbose,
            )
            if optimizer.verbose and X_ocba is not None:
                print(f"  OCBA: Adding {X_ocba.shape[0]} re-evaluation(s)")

    return X_ocba
