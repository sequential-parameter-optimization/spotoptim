"""
Analytical test functions for optimization benchmarking.

This module provides well-known analytical test functions commonly used
for evaluating and benchmarking optimization algorithms.
"""

import numpy as np


def rosenbrock(X) -> np.ndarray:
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


def ackley(X) -> np.ndarray:
    """N-dimensional Ackley function.

    The Ackley function is a widely used test function for optimization algorithms.
    It is characterized by a nearly flat outer region and a large hole at the center.
    The function is multimodal with many local minima but only one global minimum.

    Mathematical formula:
        f(X) = -a * exp(-b * sqrt(sum(x_i^2) / n)) - exp(sum(cos(c * x_i)) / n) + a + e

    where:
        - a = 20 (default)
        - b = 0.2 (default)
        - c = 2π (default)
        - e = exp(1) ≈ 2.71828
        - n = number of dimensions

    Args:
        X (array-like): Input points with shape (n_samples, n_features) or (n_features,).
            Can be a 1D array for a single point or 2D array for multiple points.

    Returns:
        np.ndarray: Function values at the input points with shape (n_samples,).

    Note:
        - Global minimum: f(0, 0, ..., 0) = 0
        - Typical search domain: [-32.768, 32.768]^N
        - Characteristics: Non-convex, multimodal, separable

    Examples:
        Single point evaluation at global minimum:

        >>> from spotoptim.function import ackley
        >>> import numpy as np
        >>> X = np.array([0.0, 0.0, 0.0])
        >>> ackley(X)
        array([0.])

        Multiple points evaluation:

        >>> X = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]])
        >>> result = ackley(X)
        >>> result[0]  # Should be close to 0
        0.0
        >>> result[1] > 0  # Should be positive
        True

    References:
        Ackley, D. H. (1987). "A connectionist machine for genetic hillclimbing".
        Kluwer Academic Publishers.
    """
    X = np.atleast_2d(X).astype(float)

    a = 20
    b = 0.2
    c = 2 * np.pi
    n_dim = X.shape[1]

    sum_sq = np.sum(X**2, axis=1)
    sum_cos = np.sum(np.cos(c * X), axis=1)

    term1 = -a * np.exp(-b * np.sqrt(sum_sq / n_dim))
    term2 = -np.exp(sum_cos / n_dim)

    return term1 + term2 + a + np.exp(1)


def michalewicz(X, m: int = 10) -> np.ndarray:
    """N-dimensional Michalewicz function.

    The Michalewicz function is a multimodal test function with steep ridges and valleys.
    The parameter m defines the steepness of the valleys and ridges. Larger values of m
    result in more difficult search problems. The number of local minima increases
    exponentially with the dimension.

    Mathematical formula:
        f(X) = -sum_{i=1}^{n} sin(x_i) * [sin(i * x_i^2 / π)]^(2m)

    Args:
        X (array-like): Input points with shape (n_samples, n_features) or (n_features,).
            Can be a 1D array for a single point or 2D array for multiple points.
        m (int, optional): Steepness parameter. Higher values make the function more
            difficult to optimize. Defaults to 10.

    Returns:
        np.ndarray: Function values at the input points with shape (n_samples,).

    Note:
        - Global minimum depends on dimension:
            - 2D: f(2.20, 1.57) ≈ -1.8013
            - 5D: f ≈ -4.687658
            - 10D: f ≈ -9.66015
        - Typical search domain: [0, π]^N
        - Characteristics: Non-convex, multimodal, non-separable

    Examples:
        Single point evaluation:

        >>> from spotoptim.function import michalewicz
        >>> import numpy as np
        >>> X = np.array([2.20, 1.57])
        >>> result = michalewicz(X)
        >>> result[0]  # Should be close to -1.8013
        -1.801303...

        Multiple points evaluation:

        >>> X = np.array([[2.20, 1.57], [1.0, 1.0]])
        >>> michalewicz(X)
        array([-1.8013..., -1.4508...])

        Using different steepness parameter:

        >>> X = np.array([2.20, 1.57])
        >>> michalewicz(X, m=5)
        array([-1.6862...])

    References:
        Michalewicz, Z. (1996). "Genetic Algorithms + Data Structures = Evolution Programs".
        Springer-Verlag.
    """
    X = np.atleast_2d(X).astype(float)

    i = np.arange(1, X.shape[1] + 1)
    # Broadcasting: (n_samples, n_features)
    result = -np.sum(np.sin(X) * (np.sin(i * X**2 / np.pi)) ** (2 * m), axis=1)

    return result
