"""
Analytical test functions for optimization benchmarking.

This module provides well-known analytical test functions commonly used
for evaluating and benchmarking optimization algorithms.
"""

import numpy as np


def rosenbrock(X):
    """N-dimensional Rosenbrock function.

    The Rosenbrock function is a classic test function for optimization algorithms.
    It is characterized by a long, narrow, parabolic-shaped valley. The global
    minimum is inside the valley and is hard to find for many algorithms.

    For the 2D case:
        f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2

    The generalized form for N dimensions:
        f(X) = sum_{i=1}^{N-1} [100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2]

    Args:
        X (array-like): Input points with shape (n_samples, n_features) or (n_features,).
            Can be a 1D array for a single point or 2D array for multiple points.

    Returns:
        np.ndarray: Function values at the input points with shape (n_samples,).

    Raises:
        ValueError: If X has fewer than 2 dimensions.

    Note:
        - Global minimum: f(1, 1, ..., 1) = 0
        - Typical search domain: [-5, 10]^N or [-2, 2]^N
        - Characteristics: Non-convex, unimodal

    Examples:
        Single point evaluation:

        >>> from spotoptim.function import rosenbrock
        >>> import numpy as np
        >>> X = np.array([1.0, 1.0])
        >>> rosenbrock(X)
        array([0.])

        Multiple points evaluation:

        >>> X = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        >>> rosenbrock(X)
        array([1.00e+00, 0.00e+00, 3.06e+01])

    References:
        Rosenbrock, H.H. (1960). "An automatic method for finding the
        greatest or least value of a function". The Computer Journal.
        3 (3): 175–184.
    """
    X = np.atleast_2d(X).astype(float)

    if X.shape[1] < 2:
        raise ValueError(
            f"Rosenbrock function requires at least 2 dimensions, got {X.shape[1]}"
        )

    # For 2D case (optimized)
    if X.shape[1] == 2:
        x, y = X[:, 0], X[:, 1]
        return (1 - x) ** 2 + 100 * (y - x**2) ** 2

    # For N-dimensional case
    result = np.zeros(X.shape[0], dtype=float)
    for i in range(X.shape[1] - 1):
        x_i = X[:, i]
        x_i_plus_1 = X[:, i + 1]
        result += 100 * (x_i_plus_1 - x_i**2) ** 2 + (1 - x_i) ** 2

    return result
