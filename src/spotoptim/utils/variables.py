# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Variable type detection, bounds processing, and factor mapping utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from spotoptim.core.protocol import SpotOptimProtocol


def detect_var_type(optimizer: SpotOptimProtocol) -> list:
    """Auto-detect variable types based on factor mappings.

    Args:
        optimizer: SpotOptim instance.

    Returns:
        list: List of variable types ('factor' or 'float') for each dimension.
    """
    return [
        "factor" if i in optimizer._factor_maps else "float"
        for i in range(optimizer.n_dim)
    ]


def modify_bounds_based_on_var_type(optimizer: SpotOptimProtocol) -> None:
    """Modify bounds based on variable types.

    Adjusts bounds for each dimension according to its var_type:
        * 'int': Ensures bounds are integers (ceiling for lower, floor for upper)
        * 'factor': Bounds already set to (0, n_levels-1) by process_factor_bounds
        * 'float': Explicitly converts bounds to float

    Args:
        optimizer: SpotOptim instance.

    Raises:
        ValueError: If an unsupported var_type is encountered.
    """
    for i, vtype in enumerate(optimizer.var_type):
        if vtype == "int":
            lower = int(np.ceil(optimizer.bounds[i][0]))
            upper = int(np.floor(optimizer.bounds[i][1]))
            optimizer.bounds[i] = (lower, upper)
        elif vtype == "factor":
            lower = int(optimizer.bounds[i][0])
            upper = int(optimizer.bounds[i][1])
            optimizer.bounds[i] = (lower, upper)
        elif vtype == "float":
            lower = float(optimizer.bounds[i][0])
            upper = float(optimizer.bounds[i][1])
            optimizer.bounds[i] = (lower, upper)
        else:
            raise ValueError(
                f"Unsupported var_type '{vtype}' at dimension {i}. "
                f"Supported types are 'float', 'int', 'factor'."
            )


def repair_non_numeric(X: np.ndarray, var_type: List[str]) -> np.ndarray:
    """Round non-numeric values to integers based on variable type.

    Args:
        X (ndarray): X array with values to potentially round.
        var_type (list of str): List with type information for each dimension.

    Returns:
        ndarray: X array with non-continuous values rounded to integers.
    """
    mask = np.isin(var_type, ["float", "float"], invert=True)
    X[:, mask] = np.around(X[:, mask])
    return X


def internal_var_type(optimizer: SpotOptimProtocol) -> List[str]:
    """Variable types effective in the internal (transformed) search space.

    Integer dimensions with an active transform are searched *continuously* in
    transformed space and only rounded after the inverse transform, in natural
    space (see ``repair_natural_X``). Rounding their internal representation
    would collapse e.g. a ``(10, 5000, "log10")`` dimension to the decade
    exponents ``{10, 100, 1000, 10000}`` — four values, the last of which
    exceeds the declared upper bound (issue #87). Factor dimensions keep their
    integer-index representation regardless of transforms.

    Args:
        optimizer: SpotOptim instance.

    Returns:
        list of str: Per-dimension types for internal-space repair —
        ``"float"`` for transformed integer dimensions, the declared type
        otherwise.
    """
    return [
        "float" if (vtype == "int" and trans is not None) else vtype
        for vtype, trans in zip(optimizer.var_type, optimizer.var_trans)
    ]


def repair_natural_X(optimizer: SpotOptimProtocol, X: np.ndarray) -> np.ndarray:
    """Enforce integrality and declared bounds in natural (original) space.

    Companion to ``repair_non_numeric`` for the *natural* scale: integer
    dimensions are rounded to the nearest integer, and integer dimensions with
    an active transform are additionally clipped to their declared natural
    bounds — the inverse transform of a continuous internal proposal can land
    marginally outside them (e.g. ``10 ** 3.69897 = 4999.99...`` rounds to
    ``5000``, while floating-point error can also yield ``5000.0000001``).
    Float and factor dimensions are left untouched (factor indices are
    integral already and are mapped to labels later).

    Handles both full-dimensional input (e.g. ``evaluate_function`` after
    ``to_all_dim``) and reduced-dimensional input (the storage paths) by
    matching the column count against the optimizer's reduced and full
    metadata. Input with an unrecognized width is returned unchanged.

    Args:
        optimizer: SpotOptim instance.
        X (ndarray): Points in natural scale, shape (n_samples, n_features)
            or (n_features,).

    Returns:
        ndarray: Repaired copy of ``X`` (same shape as the input).
    """
    X_in = np.asarray(X, dtype=float)
    one_dim = X_in.ndim == 1
    X_rep = np.atleast_2d(X_in).copy()
    n_cols = X_rep.shape[1]

    nat_lower = getattr(optimizer, "_original_lower", None)
    nat_upper = getattr(optimizer, "_original_upper", None)
    if nat_lower is None or nat_upper is None:
        return X_in

    if n_cols == len(optimizer.var_type):
        var_type = optimizer.var_type
        var_trans = optimizer.var_trans
        if getattr(optimizer, "red_dim", False) and len(nat_lower) != n_cols:
            ident = optimizer.ident
            nat_lower = nat_lower[~ident]
            nat_upper = nat_upper[~ident]
    elif n_cols == len(getattr(optimizer, "all_var_type", [])):
        var_type = optimizer.all_var_type
        var_trans = optimizer.all_var_trans
    else:
        # Unknown column layout: do not guess, return the input unchanged.
        return X_in

    for i, (vtype, trans) in enumerate(zip(var_type, var_trans)):
        if vtype != "int":
            continue
        X_rep[:, i] = np.around(X_rep[:, i])
        if trans is not None:
            X_rep[:, i] = np.clip(X_rep[:, i], nat_lower[i], nat_upper[i])

    return X_rep[0] if one_dim else X_rep


def handle_default_var_trans(optimizer: SpotOptimProtocol) -> None:
    """Handle default variable transformations.

    Sets var_trans to a list of None values if not specified, or normalizes
    transformation names by converting 'id', 'None', or None to None.

    Args:
        optimizer: SpotOptim instance.

    Raises:
        ValueError: If var_trans length doesn't match n_dim.
    """
    if optimizer.var_trans is None:
        optimizer.var_trans = [None] * optimizer.n_dim
    else:
        optimizer.var_trans = [
            None if (t is None or t == "id" or t == "None") else t
            for t in optimizer.var_trans
        ]

    if len(optimizer.var_trans) != optimizer.n_dim:
        raise ValueError(
            f"Length of var_trans ({len(optimizer.var_trans)}) must match "
            f"number of dimensions ({optimizer.n_dim})"
        )


def process_factor_bounds(optimizer: SpotOptimProtocol) -> None:
    """Process bounds to handle factor variables.

    For dimensions with tuple bounds (factor variables), creates internal
    integer mappings and replaces bounds with (0, n_levels-1).
    Stores mappings in optimizer._factor_maps: {dim_idx: {int_val: str_val}}

    Args:
        optimizer: SpotOptim instance.

    Raises:
        ValueError: If bounds are invalidly formatted.
    """
    processed_bounds = []

    for dim_idx, bound in enumerate(optimizer.bounds):
        if isinstance(bound, (tuple, list)) and len(bound) >= 1:
            # Check if this is a factor variable (contains strings)
            if all(isinstance(v, str) for v in bound) and len(bound) > 0:
                # Factor variable: create integer mapping
                factor_levels = list(bound)
                n_levels = len(factor_levels)

                # Create mapping: {0: "level1", 1: "level2", ...}
                optimizer._factor_maps[dim_idx] = {
                    i: level for i, level in enumerate(factor_levels)
                }

                # Replace with integer bounds (use Python int, not numpy types)
                processed_bounds.append((int(0), int(n_levels - 1)))

                if optimizer.verbose:
                    print(f"Factor variable at dimension {dim_idx}:")
                    print(f"  Levels: {factor_levels}")
                    print(f"  Mapped to integers: 0 to {n_levels - 1}")
            elif len(bound) == 2 and all(
                isinstance(v, (int, float, np.integer, np.floating)) for v in bound
            ):
                # Numeric bound tuple (accepts Python and numpy numeric types)
                low, high = float(bound[0]), float(bound[1])

                # Convert to int if both are integer-valued
                if low.is_integer() and high.is_integer():
                    low, high = int(low), int(high)

                processed_bounds.append((low, high))
            else:
                raise ValueError(
                    f"Invalid bound at dimension {dim_idx}: {bound}. "
                    f"Expected either (lower, upper) for numeric variables or "
                    f"tuple of strings for factor variables."
                )
        else:
            raise ValueError(
                f"Invalid bound at dimension {dim_idx}: {bound}. "
                f"Expected a tuple/list with at least 1 element."
            )

    # Update bounds with processed values
    optimizer.bounds = processed_bounds


def map_to_factor_values(optimizer: SpotOptimProtocol, X: np.ndarray) -> np.ndarray:
    """Map internal integer factor values back to string labels.

    For factor variables, converts integer indices back to original string values.
    Other variable types remain unchanged.

    Args:
        optimizer: SpotOptim instance.
        X (ndarray): Design points with integer values for factors,
            shape (n_samples, n_features).

    Returns:
        ndarray: Design points with factor integers replaced by string labels.
    """
    if not optimizer._factor_maps:
        return X

    X = np.atleast_2d(X)
    # Create object array to hold mixed types (strings and numbers)
    X_mapped = np.empty(X.shape, dtype=object)
    X_mapped[:] = X  # Copy numeric values

    for dim_idx, mapping in optimizer._factor_maps.items():
        # Check if already mapped (strings) or needs mapping (numeric)
        col_values = X[:, dim_idx]

        # If already strings, keep them
        if isinstance(col_values[0], str):
            continue

        # Round to nearest integer and map to string
        int_values = np.round(col_values).astype(int)
        # Clip to valid range
        int_values = np.clip(int_values, 0, len(mapping) - 1)
        # Map to strings
        for i, val in enumerate(int_values):
            X_mapped[i, dim_idx] = mapping[int(val)]

    return X_mapped
