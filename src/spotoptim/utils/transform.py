# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Value and array transformation utilities (log, sqrt, pow, etc.)."""

import re
from typing import Optional

import numpy as np


def transform_value(x: float, trans: Optional[str]) -> float:
    """Apply transformation to a single float value.

    Args:
        x: Value to transform
        trans: Transformation name. Can be one of 'id', 'log10', 'log', 'ln', 'sqrt',
               'exp', 'square', 'cube', 'inv', 'reciprocal', or None.
               Also supports dynamic strings like 'log(x)', 'sqrt(x)', 'pow(x, p)'.

    Returns:
        Transformed value

    Raises:
        TypeError: If x is not a float.
        ValueError: If an unknown transformation is specified.
    """
    if not isinstance(x, float):
        try:
            x = float(x)
        except (ValueError, TypeError):
            raise TypeError(
                f"transform_value expects a float, got {type(x).__name__} (value: {x})"
            )
    if trans is None or trans == "id":
        return x
    elif trans == "log10":
        return np.log10(x)
    elif trans == "log" or trans == "ln":
        return np.log(x)
    elif trans == "sqrt":
        return np.sqrt(x)
    elif trans == "exp":
        return np.exp(x)
    elif trans == "square":
        return x**2
    elif trans == "cube":
        return x**3
    elif trans == "inv" or trans == "reciprocal":
        return 1.0 / x

    # Dynamic Transformations

    if trans == "log(x)":
        return np.log(x)
    if trans == "sqrt(x)":
        return np.sqrt(x)

    m = re.match(r"pow\(x,\s*([0-9.]+)\)", trans)
    if m:
        p = float(m.group(1))
        return x**p

    m = re.match(r"pow\(([0-9.]+),\s*x\)", trans)
    if m:
        base = float(m.group(1))
        return base**x

    m = re.match(r"log\(x,\s*([0-9.]+)\)", trans)
    if m:
        base = float(m.group(1))
        return np.log(x) / np.log(base)

    raise ValueError(f"Unknown transformation: {trans}")


def inverse_transform_value(x: float, trans: Optional[str]) -> float:
    """Apply inverse transformation to a single float value.

    Args:
        x: Transformed value
        trans: Transformation name.

    Returns:
        Original value

    Raises:
        TypeError: If x is not a float.
        ValueError: If an unknown transformation is specified.
    """
    if not isinstance(x, float):
        try:
            x = float(x)
        except (ValueError, TypeError):
            raise TypeError(
                f"transform_value expects a float, got {type(x).__name__} (value: {x})"
            )
    if trans is None or trans == "id":
        return x
    elif trans == "log10":
        return 10**x
    elif trans == "log" or trans == "ln":
        return np.exp(x)
    elif trans == "sqrt":
        return x**2
    elif trans == "exp":
        return np.log(x)
    elif trans == "square":
        return np.sqrt(x)
    elif trans == "cube":
        return np.power(x, 1.0 / 3.0)
    elif trans == "inv" or trans == "reciprocal":
        return 1.0 / x

    # Dynamic Transformations (Inverses)
    if trans == "log(x)":
        return np.exp(x)
    if trans == "sqrt(x)":
        return x**2

    m = re.match(r"pow\(x,\s*([0-9.]+)\)", trans)
    if m:
        p = float(m.group(1))
        return x ** (1.0 / p)

    m = re.match(r"pow\(([0-9.]+),\s*x\)", trans)
    if m:
        base = float(m.group(1))
        return np.log(x) / np.log(base)

    m = re.match(r"log\(x,\s*([0-9.]+)\)", trans)
    if m:
        base = float(m.group(1))
        return base**x

    raise ValueError(f"Unknown transformation: {trans}")


def transform_X(optimizer, X: np.ndarray) -> np.ndarray:
    """Transform parameter array from original (natural) to internal scale.

    Args:
        optimizer: SpotOptim instance.
        X (ndarray): Array in Natural Space, shape (n_samples, n_features)

    Returns:
        ndarray: Array in Transformed Space (Full Dimension)
    """
    X_transformed = X.copy()

    # Handle 1D array
    if X.ndim == 1:
        for i, trans in enumerate(optimizer.var_trans):
            if trans is not None:
                X_transformed[i] = transform_value(X[i], trans)
        return X_transformed

    # Handle 2D array
    for i, trans in enumerate(optimizer.var_trans):
        if trans is not None:
            X_transformed[:, i] = np.array([transform_value(x, trans) for x in X[:, i]])
    return X_transformed


def inverse_transform_X(optimizer, X: np.ndarray) -> np.ndarray:
    """Transform parameter array from internal to original scale.

    Args:
        optimizer: SpotOptim instance.
        X (ndarray): Array in Transformed Space, shape (n_samples, n_features)

    Returns:
        ndarray: Array in Natural Space
    """
    X_original = X.copy()

    # Handle 1D array (single sample)
    if X.ndim == 1:
        for i, trans in enumerate(optimizer.var_trans):
            if trans is not None:
                X_original[i] = inverse_transform_value(X[i], trans)
        return X_original

    # Handle 2D array (multiple samples)
    for i, trans in enumerate(optimizer.var_trans):
        if trans is not None:
            X_original[:, i] = np.array(
                [inverse_transform_value(x, trans) for x in X[:, i]]
            )
    return X_original


def transform_bounds(optimizer) -> None:
    """Transform bounds from original to internal scale.

    Updates optimizer.bounds (and optimizer.lower, optimizer.upper) from Natural Space
    to Transformed Space.

    Args:
        optimizer: SpotOptim instance.
    """
    for i, trans in enumerate(optimizer.var_trans):
        if trans is not None:
            lower_t = transform_value(optimizer.lower[i], trans)
            upper_t = transform_value(optimizer.upper[i], trans)

            # Handle reversed bounds (e.g., reciprocal transformation)
            if lower_t > upper_t:
                optimizer.lower[i], optimizer.upper[i] = upper_t, lower_t
            else:
                optimizer.lower[i], optimizer.upper[i] = lower_t, upper_t

    # Update optimizer.bounds to reflect transformed bounds
    # Convert numpy types to Python native types (int or float based on var_type)
    optimizer.bounds = []
    for i in range(len(optimizer.lower)):
        if i < len(optimizer.var_type) and (
            optimizer.var_type[i] == "int" or optimizer.var_type[i] == "factor"
        ):
            optimizer.bounds.append((int(optimizer.lower[i]), int(optimizer.upper[i])))
        else:
            optimizer.bounds.append(
                (float(optimizer.lower[i]), float(optimizer.upper[i]))
            )
