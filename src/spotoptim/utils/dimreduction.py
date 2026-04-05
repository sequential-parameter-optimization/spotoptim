# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Dimension reduction utilities for fixed-variable elimination."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats.qmc import LatinHypercube

if TYPE_CHECKING:
    from spotoptim.core.protocol import SpotOptimProtocol


def setup_dimension_reduction(optimizer: SpotOptimProtocol) -> None:
    """Set up dimension reduction by identifying fixed dimensions.

    Identifies dimensions where lower and upper bounds are equal in Transformed Space.
    Reduces optimizer.bounds, optimizer.lower, optimizer.upper, etc., to the Mapped Space
    (active variables only).

    Args:
        optimizer: SpotOptim instance.
    """
    # Backup original values
    optimizer.all_lower = optimizer.lower.copy()
    optimizer.all_upper = optimizer.upper.copy()
    optimizer.all_var_type = optimizer.var_type.copy()
    optimizer.all_var_name = optimizer.var_name.copy()
    optimizer.all_var_trans = optimizer.var_trans.copy()

    # Identify fixed dimensions (lower == upper)
    optimizer.ident = (optimizer.upper - optimizer.lower) == 0

    # Check if any dimension is fixed
    optimizer.red_dim = optimizer.ident.any()

    if optimizer.red_dim:
        # Reduce bounds to only varying dimensions
        optimizer.lower = optimizer.lower[~optimizer.ident]
        optimizer.upper = optimizer.upper[~optimizer.ident]

        # Update dimension count
        optimizer.n_dim = optimizer.lower.size

        # Reduce variable types and names
        optimizer.var_type = [
            vtype
            for vtype, fixed in zip(optimizer.all_var_type, optimizer.ident)
            if not fixed
        ]
        optimizer.var_name = [
            vname
            for vname, fixed in zip(optimizer.all_var_name, optimizer.ident)
            if not fixed
        ]

        # Reduce transformations
        optimizer.var_trans = [
            vtrans
            for vtrans, fixed in zip(optimizer.all_var_trans, optimizer.ident)
            if not fixed
        ]

        # Update bounds list for reduced dimensions
        optimizer.bounds = []
        for i in range(optimizer.n_dim):
            if i < len(optimizer.var_type) and (
                optimizer.var_type[i] == "int" or optimizer.var_type[i] == "factor"
            ):
                optimizer.bounds.append(
                    (int(optimizer.lower[i]), int(optimizer.upper[i]))
                )
            else:
                optimizer.bounds.append(
                    (float(optimizer.lower[i]), float(optimizer.upper[i]))
                )

        # Recreate LHS sampler with reduced dimensions
        optimizer.lhs_sampler = LatinHypercube(d=optimizer.n_dim, rng=optimizer.seed)


def to_red_dim(optimizer: SpotOptimProtocol, X_full: np.ndarray) -> np.ndarray:
    """Reduce full-dimensional points to optimization space.

    Removes fixed dimensions from full-dimensional points.

    Args:
        optimizer: SpotOptim instance.
        X_full (ndarray): Points in full space, shape (n_samples, n_original_dims).

    Returns:
        ndarray: Points in reduced space, shape (n_samples, n_reduced_dims).
    """
    if not optimizer.red_dim:
        return X_full

    # Handle 1D array
    if X_full.ndim == 1:
        return X_full[~optimizer.ident]

    # Select only non-fixed dimensions (2D)
    return X_full[:, ~optimizer.ident]


def to_all_dim(optimizer: SpotOptimProtocol, X_red: np.ndarray) -> np.ndarray:
    """Expand reduced-dimensional points to full-dimensional representation.

    Restores points from the reduced optimization space to the full-dimensional
    space by inserting fixed values for constant dimensions.

    Args:
        optimizer: SpotOptim instance.
        X_red (ndarray): Points in reduced space, shape (n_samples, n_reduced_dims).

    Returns:
        ndarray: Points in full space, shape (n_samples, n_original_dims).
    """
    if not optimizer.red_dim:
        return X_red

    # Number of samples and full dimensions
    n_samples = X_red.shape[0]
    n_full_dims = len(optimizer.ident)

    # Initialize full-dimensional array
    X_full = np.zeros((n_samples, n_full_dims))

    # Track index in reduced array
    red_idx = 0

    # Fill in values dimension by dimension
    for i in range(n_full_dims):
        if optimizer.ident[i]:
            # Fixed dimension: use stored value
            X_full[:, i] = optimizer.all_lower[i]
        else:
            # Varying dimension: use value from reduced array
            X_full[:, i] = X_red[:, red_idx]
            red_idx += 1

    return X_full
