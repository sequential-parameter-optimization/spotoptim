"""
Analytical multi-objective test functions for optimization benchmarking.

This module provides well-known multi-objective analytical test functions commonly used
for evaluating and benchmarking multiobjective optimization algorithms.
"""

import numpy as np


def zdt1(X) -> np.ndarray:
    """ZDT1 multi-objective test function (2 objectives).

    ZDT1 is a classical bi-objective test problem with a convex Pareto front.
    It is one of the most widely used benchmark functions for multi-objective optimization.

    Mathematical formulation:
        f1(X) = x1
        f2(X) = g(X) * [1 - sqrt(x1 / g(X))]
        g(X) = 1 + 9 * sum(x_i for i=2 to n) / (n - 1)

    Args:
        X (array-like): Input points with shape (n_samples, n_features) or (n_features,).
            Can be a 1D array for a single point or 2D array for multiple points.
            Must have at least 2 dimensions.

    Returns:
        np.ndarray: Objective values with shape (n_samples, 2) where:
            - Column 0: f1 values
            - Column 1: f2 values

    Raises:
        ValueError: If X has fewer than 2 dimensions.

    Note:
        - Number of objectives: 2
        - Typical number of variables: 30
        - Search domain: [0, 1]^n
        - Pareto front: Convex, f1 ∈ [0, 1], f2 = 1 - sqrt(f1)
        - Characteristics: Convex, unimodal

    Examples:
        Single point evaluation:

        >>> from spotoptim.function.mo import zdt1
        >>> import numpy as np
        >>> X = np.array([0.0, 0.0, 0.0])
        >>> result = zdt1(X)
        >>> result.shape
        (1, 2)
        >>> result[0, 0]  # f1
        0.0
        >>> result[0, 1]  # f2
        1.0

        Multiple points evaluation:

        >>> X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        >>> result = zdt1(X)
        >>> result.shape
        (3, 2)

    References:
        Zitzler, E., Deb, K., & Thiele, L. (2000). "Comparison of multiobjective
        evolutionary algorithms: Empirical results." Evolutionary computation, 8(2), 173-195.
    """
    X = np.atleast_2d(X).astype(float)

    if X.shape[1] < 2:
        raise ValueError(f"ZDT1 requires at least 2 dimensions, got {X.shape[1]}")

    n = X.shape[1]
    f1 = X[:, 0]

    g = 1 + 9 * np.sum(X[:, 1:], axis=1) / (n - 1)
    f2 = g * (1 - np.sqrt(f1 / g))

    return np.column_stack([f1, f2])


def zdt2(X) -> np.ndarray:
    """ZDT2 multi-objective test function (2 objectives).

    ZDT2 is similar to ZDT1 but has a non-convex Pareto front.

    Mathematical formulation:
        f1(X) = x1
        f2(X) = g(X) * [1 - (x1 / g(X))^2]
        g(X) = 1 + 9 * sum(x_i for i=2 to n) / (n - 1)

    Args:
        X (array-like): Input points with shape (n_samples, n_features) or (n_features,).
            Can be a 1D array for a single point or 2D array for multiple points.
            Must have at least 2 dimensions.

    Returns:
        np.ndarray: Objective values with shape (n_samples, 2) where:
            - Column 0: f1 values
            - Column 1: f2 values

    Raises:
        ValueError: If X has fewer than 2 dimensions.

    Note:
        - Number of objectives: 2
        - Typical number of variables: 30
        - Search domain: [0, 1]^n
        - Pareto front: Non-convex, f1 ∈ [0, 1], f2 = 1 - f1^2
        - Characteristics: Non-convex, unimodal

    Examples:
        >>> from spotoptim.function.mo import zdt2
        >>> import numpy as np
        >>> X = np.array([0.0, 0.0, 0.0])
        >>> result = zdt2(X)
        >>> result.shape
        (1, 2)

    References:
        Zitzler, E., Deb, K., & Thiele, L. (2000). "Comparison of multiobjective
        evolutionary algorithms: Empirical results." Evolutionary computation, 8(2), 173-195.
    """
    X = np.atleast_2d(X).astype(float)

    if X.shape[1] < 2:
        raise ValueError(f"ZDT2 requires at least 2 dimensions, got {X.shape[1]}")

    n = X.shape[1]
    f1 = X[:, 0]

    g = 1 + 9 * np.sum(X[:, 1:], axis=1) / (n - 1)
    f2 = g * (1 - (f1 / g) ** 2)

    return np.column_stack([f1, f2])


def zdt3(X) -> np.ndarray:
    """ZDT3 multi-objective test function (2 objectives).

    ZDT3 has a disconnected (discontinuous) Pareto front, making it more challenging.

    Mathematical formulation:
        f1(X) = x1
        f2(X) = g(X) * [1 - sqrt(x1 / g(X)) - (x1 / g(X)) * sin(10 * π * x1)]
        g(X) = 1 + 9 * sum(x_i for i=2 to n) / (n - 1)

    Args:
        X (array-like): Input points with shape (n_samples, n_features) or (n_features,).
            Can be a 1D array for a single point or 2D array for multiple points.
            Must have at least 2 dimensions.

    Returns:
        np.ndarray: Objective values with shape (n_samples, 2) where:
            - Column 0: f1 values
            - Column 1: f2 values

    Raises:
        ValueError: If X has fewer than 2 dimensions.

    Note:
        - Number of objectives: 2
        - Typical number of variables: 30
        - Search domain: [0, 1]^n
        - Pareto front: Disconnected (5 separate regions)
        - Characteristics: Discontinuous, multimodal

    Examples:
        >>> from spotoptim.function.mo import zdt3
        >>> import numpy as np
        >>> X = np.array([0.0, 0.0, 0.0])
        >>> result = zdt3(X)
        >>> result.shape
        (1, 2)

    References:
        Zitzler, E., Deb, K., & Thiele, L. (2000). "Comparison of multiobjective
        evolutionary algorithms: Empirical results." Evolutionary computation, 8(2), 173-195.
    """
    X = np.atleast_2d(X).astype(float)

    if X.shape[1] < 2:
        raise ValueError(f"ZDT3 requires at least 2 dimensions, got {X.shape[1]}")

    n = X.shape[1]
    f1 = X[:, 0]

    g = 1 + 9 * np.sum(X[:, 1:], axis=1) / (n - 1)
    f2 = g * (1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1))

    return np.column_stack([f1, f2])


def zdt4(X) -> np.ndarray:
    """ZDT4 multi-objective test function (2 objectives).

    ZDT4 has 21^9 local Pareto fronts, testing the algorithm's ability to deal with multimodality.

    Mathematical formulation:
        f1(X) = x1
        f2(X) = g(X) * [1 - sqrt(x1 / g(X))]
        g(X) = 1 + 10 * (n - 1) + sum(x_i^2 - 10 * cos(4 * π * x_i) for i=2 to n)

    Args:
        X (array-like): Input points with shape (n_samples, n_features) or (n_features,).
            Can be a 1D array for a single point or 2D array for multiple points.
            Must have at least 2 dimensions.

    Returns:
        np.ndarray: Objective values with shape (n_samples, 2) where:
            - Column 0: f1 values
            - Column 1: f2 values

    Raises:
        ValueError: If X has fewer than 2 dimensions.

    Note:
        - Number of objectives: 2
        - Typical number of variables: 10
        - Search domain: x1 ∈ [0, 1], x_i ∈ [-5, 5] for i = 2, ..., n
        - Pareto front: Convex, same as ZDT1
        - Characteristics: Multimodal (many local fronts)

    Examples:
        >>> from spotoptim.function.mo import zdt4
        >>> import numpy as np
        >>> X = np.array([0.5, 0.0, 0.0])
        >>> result = zdt4(X)
        >>> result.shape
        (1, 2)

    References:
        Zitzler, E., Deb, K., & Thiele, L. (2000). "Comparison of multiobjective
        evolutionary algorithms: Empirical results." Evolutionary computation, 8(2), 173-195.
    """
    X = np.atleast_2d(X).astype(float)

    if X.shape[1] < 2:
        raise ValueError(f"ZDT4 requires at least 2 dimensions, got {X.shape[1]}")

    n = X.shape[1]
    f1 = X[:, 0]

    g = (
        1
        + 10 * (n - 1)
        + np.sum(X[:, 1:] ** 2 - 10 * np.cos(4 * np.pi * X[:, 1:]), axis=1)
    )
    f2 = g * (1 - np.sqrt(f1 / g))

    return np.column_stack([f1, f2])


def zdt6(X) -> np.ndarray:
    """ZDT6 multi-objective test function (2 objectives).

    ZDT6 has a non-uniform search space with a non-convex Pareto front and
    low density of solutions near the Pareto front.

    Mathematical formulation:
        f1(X) = 1 - exp(-4 * x1) * sin^6(6 * π * x1)
        f2(X) = g(X) * [1 - (f1 / g(X))^2]
        g(X) = 1 + 9 * [sum(x_i for i=2 to n) / (n - 1)]^0.25

    Args:
        X (array-like): Input points with shape (n_samples, n_features) or (n_features,).
            Can be a 1D array for a single point or 2D array for multiple points.
            Must have at least 2 dimensions.

    Returns:
        np.ndarray: Objective values with shape (n_samples, 2) where:
            - Column 0: f1 values
            - Column 1: f2 values

    Raises:
        ValueError: If X has fewer than 2 dimensions.

    Note:
        - Number of objectives: 2
        - Typical number of variables: 10
        - Search domain: [0, 1]^n
        - Pareto front: Non-convex, non-uniform density
        - Characteristics: Non-uniform, biased search space

    Examples:
        >>> from spotoptim.function.mo import zdt6
        >>> import numpy as np
        >>> X = np.array([0.5, 0.5, 0.5])
        >>> result = zdt6(X)
        >>> result.shape
        (1, 2)

    References:
        Zitzler, E., Deb, K., & Thiele, L. (2000). "Comparison of multiobjective
        evolutionary algorithms: Empirical results." Evolutionary computation, 8(2), 173-195.
    """
    X = np.atleast_2d(X).astype(float)

    if X.shape[1] < 2:
        raise ValueError(f"ZDT6 requires at least 2 dimensions, got {X.shape[1]}")

    n = X.shape[1]
    f1 = 1 - np.exp(-4 * X[:, 0]) * (np.sin(6 * np.pi * X[:, 0]) ** 6)

    g = 1 + 9 * ((np.sum(X[:, 1:], axis=1) / (n - 1)) ** 0.25)
    f2 = g * (1 - (f1 / g) ** 2)

    return np.column_stack([f1, f2])


def dtlz1(X, n_obj: int = 3) -> np.ndarray:
    """DTLZ1 multi-objective test function (scalable objectives).

    DTLZ1 is a scalable test problem with a linear Pareto front. It has
    (11^k - 1) local Pareto fronts where k = n - n_obj + 1.

    Mathematical formulation:
        f_i(X) = 0.5 * x1 * ... * x_{M-i} * [1 + g(X)] for i = 1, ..., M-1
        f_M(X) = 0.5 * [1 - x_{M-1}] * [1 + g(X)]
        g(X) = 100 * [k + sum((x_i - 0.5)^2 - cos(20π(x_i - 0.5)) for i in X_M)]

    Args:
        X (array-like): Input points with shape (n_samples, n_features) or (n_features,).
            Can be a 1D array for a single point or 2D array for multiple points.
        n_obj (int, optional): Number of objectives. Defaults to 3.
            Must be at least 2 and at most n_features.

    Returns:
        np.ndarray: Objective values with shape (n_samples, n_obj).

    Raises:
        ValueError: If n_obj is invalid or X has insufficient dimensions.

    Note:
        - Number of objectives: Scalable (typically 3)
        - Typical number of variables: n_obj + k - 1 (often k = 5, so n = 7 for 3 objectives)
        - Search domain: [0, 1]^n
        - Pareto front: Linear hyperplane
        - Characteristics: Multimodal, many local fronts

    Examples:
        >>> from spotoptim.function.mo import dtlz1
        >>> import numpy as np
        >>> X = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        >>> result = dtlz1(X, n_obj=3)
        >>> result.shape
        (1, 3)

    References:
        Deb, K., Thiele, L., Laumanns, M., & Zitzler, E. (2005). "Scalable test problems
        for evolutionary multiobjective optimization." In Evolutionary multiobjective
        optimization (pp. 105-145). Springer.
    """
    X = np.atleast_2d(X).astype(float)
    n = X.shape[1]

    if n_obj < 2:
        raise ValueError(f"Number of objectives must be at least 2, got {n_obj}")
    if n_obj > n:
        raise ValueError(
            f"Number of objectives ({n_obj}) cannot exceed number of variables ({n})"
        )

    k = n - n_obj + 1
    X_m = X[:, n_obj - 1 :]

    g = 100 * (k + np.sum((X_m - 0.5) ** 2 - np.cos(20 * np.pi * (X_m - 0.5)), axis=1))

    f = np.zeros((X.shape[0], n_obj))

    for i in range(n_obj):
        f[:, i] = 0.5 * (1 + g)
        for j in range(n_obj - i - 1):
            f[:, i] *= X[:, j]
        if i > 0:
            f[:, i] *= 1 - X[:, n_obj - i - 1]

    return f


def dtlz2(X, n_obj: int = 3) -> np.ndarray:
    """DTLZ2 multi-objective test function (scalable objectives).

    DTLZ2 is a scalable test problem with a concave Pareto front (a unit sphere).

    Mathematical formulation:
        f_i(X) = [1 + g(X)] * cos(x1 * π/2) * ... * cos(x_{M-i} * π/2) * sin(x_{M-i+1} * π/2)
        f_M(X) = [1 + g(X)] * sin(x1 * π/2)
        g(X) = sum((x_i - 0.5)^2 for i in X_M)

    Args:
        X (array-like): Input points with shape (n_samples, n_features) or (n_features,).
            Can be a 1D array for a single point or 2D array for multiple points.
        n_obj (int, optional): Number of objectives. Defaults to 3.
            Must be at least 2 and at most n_features.

    Returns:
        np.ndarray: Objective values with shape (n_samples, n_obj).

    Raises:
        ValueError: If n_obj is invalid or X has insufficient dimensions.

    Note:
        - Number of objectives: Scalable (typically 3)
        - Typical number of variables: n_obj + k - 1 (often k = 10, so n = 12 for 3 objectives)
        - Search domain: [0, 1]^n
        - Pareto front: Concave (sphere: sum(f_i^2) = 1)
        - Characteristics: Concave, unimodal

    Examples:
        >>> from spotoptim.function.mo import dtlz2
        >>> import numpy as np
        >>> X = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        >>> result = dtlz2(X, n_obj=3)
        >>> result.shape
        (1, 3)

    References:
        Deb, K., Thiele, L., Laumanns, M., & Zitzler, E. (2005). "Scalable test problems
        for evolutionary multiobjective optimization." In Evolutionary multiobjective
        optimization (pp. 105-145). Springer.
    """
    X = np.atleast_2d(X).astype(float)
    n = X.shape[1]

    if n_obj < 2:
        raise ValueError(f"Number of objectives must be at least 2, got {n_obj}")
    if n_obj > n:
        raise ValueError(
            f"Number of objectives ({n_obj}) cannot exceed number of variables ({n})"
        )

    X_m = X[:, n_obj - 1 :]
    g = np.sum((X_m - 0.5) ** 2, axis=1)

    f = np.zeros((X.shape[0], n_obj))

    for i in range(n_obj):
        f[:, i] = 1 + g

        for j in range(n_obj - i - 1):
            f[:, i] *= np.cos(X[:, j] * np.pi / 2)

        if i > 0:
            f[:, i] *= np.sin(X[:, n_obj - i - 1] * np.pi / 2)

    return f


def schaffer_n1(X) -> np.ndarray:
    """Schaffer N1 multi-objective test function (2 objectives).

    Schaffer N1 is a simple bi-objective problem with a convex Pareto front.
    It is one of the earliest multi-objective test functions.

    Mathematical formulation:
        f1(X) = x^2
        f2(X) = (x - 2)^2

    Args:
        X (array-like): Input points with shape (n_samples, n_features) or (n_features,).
            Can be a 1D array for a single point or 2D array for multiple points.
            This function uses only the first variable.

    Returns:
        np.ndarray: Objective values with shape (n_samples, 2) where:
            - Column 0: f1 values
            - Column 1: f2 values

    Note:
        - Number of objectives: 2
        - Number of variables: 1 (only first variable is used)
        - Search domain: [-10, 10] or [-A, A]
        - Pareto front: x ∈ [0, 2]
        - Characteristics: Convex, simple, unimodal

    Examples:
        >>> from spotoptim.function.mo import schaffer_n1
        >>> import numpy as np
        >>> X = np.array([0.0])
        >>> result = schaffer_n1(X)
        >>> result.shape
        (1, 2)
        >>> result[0]
        array([0., 4.])

        >>> X = np.array([[0.0], [1.0], [2.0]])
        >>> result = schaffer_n1(X)
        >>> result.shape
        (3, 2)

    References:
        Schaffer, J. D. (1985). "Multiple objective optimization with vector evaluated
        genetic algorithms." In Proceedings of the 1st international Conference on
        Genetic Algorithms (pp. 93-100).
    """
    X = np.atleast_2d(X).astype(float)
    x = X[:, 0]

    f1 = x**2
    f2 = (x - 2) ** 2

    return np.column_stack([f1, f2])


def fonseca_fleming(X) -> np.ndarray:
    """Fonseca-Fleming multi-objective test function (2 objectives).

    The Fonseca-Fleming function is a classical bi-objective problem with a
    concave Pareto front. The difficulty increases with the number of variables.

    Mathematical formulation:
        f1(X) = 1 - exp(-sum((x_i - 1/sqrt(n))^2 for i=1 to n))
        f2(X) = 1 - exp(-sum((x_i + 1/sqrt(n))^2 for i=1 to n))

    Args:
        X (array-like): Input points with shape (n_samples, n_features) or (n_features,).
            Can be a 1D array for a single point or 2D array for multiple points.

    Returns:
        np.ndarray: Objective values with shape (n_samples, 2) where:
            - Column 0: f1 values
            - Column 1: f2 values

    Note:
        - Number of objectives: 2
        - Typical number of variables: 2-10
        - Search domain: [-4, 4]^n
        - Pareto front: Concave
        - Characteristics: Concave, symmetric

    Examples:
        >>> from spotoptim.function.mo import fonseca_fleming
        >>> import numpy as np
        >>> X = np.array([0.0, 0.0])
        >>> result = fonseca_fleming(X)
        >>> result.shape
        (1, 2)

        >>> X = np.array([[0.0, 0.0], [1.0, 1.0]])
        >>> result = fonseca_fleming(X)
        >>> result.shape
        (2, 2)

    References:
        Fonseca, C. M., & Fleming, P. J. (1995). "An overview of evolutionary algorithms
        in multiobjective optimization." Evolutionary computation, 3(1), 1-16.
    """
    X = np.atleast_2d(X).astype(float)
    n = X.shape[1]

    sqrt_n_inv = 1.0 / np.sqrt(n)

    f1 = 1 - np.exp(-np.sum((X - sqrt_n_inv) ** 2, axis=1))
    f2 = 1 - np.exp(-np.sum((X + sqrt_n_inv) ** 2, axis=1))

    return np.column_stack([f1, f2])


def kursawe(X) -> np.ndarray:
    """Kursawe multi-objective test function (2 objectives, minimization).

    The Kursawe function is a classic bi-objective minimization benchmark with a
    non-convex, disconnected Pareto front, often used to test an optimizer's ability
    to maintain diversity and avoid getting trapped in local fronts.

    Mathematical formulation:
        f1(X) = sum(-10 * exp(-0.2 * sqrt(x_i^2 + x_{i+1}^2)) for i=1 to n-1)
        f2(X) = sum(|x_i|^0.8 + 5 * sin(x_i^3) for i=1 to n)

    Args:
        X (array-like): Input points with shape (n_samples, n_features) or (n_features,).
            Can be a 1D array for a single point or 2D array for multiple points.
            Must have at least 2 dimensions.

    Returns:
        np.ndarray: Objective values with shape (n_samples, 2) where:
            - Column 0: f1 values
            - Column 1: f2 values

    Raises:
        ValueError: If X has fewer than 2 dimensions.

    Note:
        - Number of objectives: 2
        - Typical number of variables: 3
        - Search domain: [-5, 5]^n
        - Pareto front: Disconnected
        - Characteristics: Non-convex, disconnected, multimodal

    Examples:
        >>> from spotoptim.function.mo import kursawe
        >>> import numpy as np
        >>> X = np.array([0.0, 0.0, 0.0])
        >>> result = kursawe(X)
        >>> result.shape
        (1, 2)

        >>> X = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        >>> result = kursawe(X)
        >>> result.shape
        (2, 2)

    References:
        Kursawe, F. (1991). "A variant of evolution strategies for vector optimization."
        In International Conference on Parallel Problem Solving from Nature (pp. 193-197).
        Springer.
    """
    X = np.atleast_2d(X).astype(float)

    if X.shape[1] < 2:
        raise ValueError(f"Kursawe requires at least 2 dimensions, got {X.shape[1]}")

    f1 = np.sum(-10 * np.exp(-0.2 * np.sqrt(X[:, :-1] ** 2 + X[:, 1:] ** 2)), axis=1)
    f2 = np.sum(np.abs(X) ** 0.8 + 5 * np.sin(X**3), axis=1)

    return np.column_stack([f1, f2])


def mo_conv2_min(X) -> np.ndarray:
    """Convex bi-objective minimization test function (2 objectives).

    A smooth, convex two-objective problem on [0, 1]^2:
        f1(x, y) = x^2 + y^2
        f2(x, y) = (x - 1)^2 + (y - 1)^2

    Properties:
        - Domain: [0, 1]^2
        - Objectives: minimize both f1 and f2
        - Ideal points: (0, 0) for f1; (1, 1) for f2
        - Pareto set: line x = y in [0, 1]
        - Pareto front: convex quadratic trade-off f1 = 2t^2, f2 = 2(1 - t)^2, t ∈ [0, 1]

    Args:
        X (array-like): Input points with shape (n_samples, n_features) or (n_features,).
            Can be a 1D array for a single point or 2D array for multiple points.
            Must have exactly 2 dimensions.

    Returns:
        np.ndarray: Objective values with shape (n_samples, 2) where:
            - Column 0: f1 values (to be minimized)
            - Column 1: f2 values (to be minimized)

    Raises:
        ValueError: If X does not have exactly 2 dimensions.

    Note:
        - Number of objectives: 2
        - Number of variables: 2
        - Search domain: [0, 1]^2
        - Ideal points: (0, 0) for f1, (1, 1) for f2
        - Pareto front: Convex, quadratic
        - Problem type: Minimization
        - Characteristics: Convex, smooth, bounded

    Examples:
        Single point evaluation:

        >>> from spotoptim.function.mo import mo_conv2_min
        >>> import numpy as np
        >>> X = np.array([0.0, 0.0])
        >>> result = mo_conv2_min(X)
        >>> result.shape
        (1, 2)
        >>> result[0]  # f1 minimum
        array([0., 2.])

        >>> X = np.array([1.0, 1.0])
        >>> result = mo_conv2_min(X)
        >>> result[0]  # f2 minimum
        array([2., 0.])

        Multiple points evaluation:

        >>> X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        >>> result = mo_conv2_min(X)
        >>> result.shape
        (3, 2)
        >>> result[1]  # Pareto front
        array([0.5, 0.5])
    """
    X = np.atleast_2d(X).astype(float)

    if X.shape[1] != 2:
        raise ValueError(
            f"mo_conv2_min requires exactly 2 dimensions, got {X.shape[1]}"
        )

    x = X[:, 0]
    y = X[:, 1]

    # Objective 1: f1(x, y) = x^2 + y^2
    f1 = x**2 + y**2

    # Objective 2: f2(x, y) = (x-1)^2 + (y-1)^2
    f2 = (x - 1) ** 2 + (y - 1) ** 2

    return np.column_stack([f1, f2])


def mo_conv2_max(X) -> np.ndarray:
    """Convex bi-objective maximization test function (2 objectives).

    A smooth, convex two-objective maximization problem on [0, 1]^2 using flipped
    versions of the minimization objectives:
        f1(x, y) = 2 - (x^2 + y^2)
        f2(x, y) = 2 - ((1 - x)^2 + (1 - y)^2)

    Properties:
        - Domain: [0, 1]^2
        - Objectives: maximize both f1 and f2
        - Ideal points: (0, 0) for f1 (gives f1=2); (1, 1) for f2 (gives f2=2)
        - Pareto set: line x = y in [0, 1]
        - Pareto front: convex quadratic trade-off f1 = 2 - 2t^2, f2 = 2 - 2(1 - t)^2, t ∈ [0, 1]

    Args:
        X (array-like): Input points with shape (n_samples, n_features) or (n_features,).
            Can be a 1D array for a single point or 2D array for multiple points.
            Must have exactly 2 dimensions.

    Returns:
        np.ndarray: Objective values with shape (n_samples, 2) where:
            - Column 0: f1 values (to be maximized)
            - Column 1: f2 values (to be maximized)

    Raises:
        ValueError: If X does not have exactly 2 dimensions.

    Note:
        - Number of objectives: 2
        - Number of variables: 2
        - Search domain: [0, 1]^2
        - Ideal points: (1, 1) for f1, (0, 0) for f2
        - Pareto front: Convex, quadratic
        - Problem type: Maximization
        - Characteristics: Convex, smooth, bounded

    Examples:
        Single point evaluation:

        >>> from spotoptim.function.mo import mo_conv2_max
        >>> import numpy as np
        >>> X = np.array([0.0, 0.0])
        >>> result = mo_conv2_max(X)
        >>> result.shape
        (1, 2)
        >>> result[0]  # f1 maximum
        array([0., 2.])

        >>> X = np.array([1.0, 1.0])
        >>> result = mo_conv2_max(X)
        >>> result[0]  # f2 maximum
        array([2., 0.])

        Multiple points evaluation:

        >>> X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        >>> result = mo_conv2_min(X)
        >>> result.shape
        (3, 2)
        >>> result[1]  # Pareto front
        array([0.5, 0.5])
    """
    X = np.atleast_2d(X).astype(float)

    if X.shape[1] != 2:
        raise ValueError(
            f"mo_conv2_min requires exactly 2 dimensions, got {X.shape[1]}"
        )

    x = X[:, 0]
    y = X[:, 1]

    # Objective 1: f1(x, y) = x^2 + y^2
    f1 = 2.0 - (x**2 + y**2)

    # Objective 2: f2(x, y) = (1-x)^2 + (1-y)^2
    f2 = 2.0 - ((1 - x) ** 2 + (1 - y) ** 2)

    return np.column_stack([f1, f2])


def conversion_pred(X) -> np.ndarray:
    """
    Compute conversion predictions for each row in the input array.

    Args:
        X (np.ndarray): 2D array where each row is a configuration.

    Returns:
        np.ndarray: 1D array of conversion predictions.

    Examples:
        >>> import numpy as np
        >>> from spotoptim.function.mo import conversion_pred
        >>> # Example input data
        >>> X = np.array([[1, 2, 3], [4, 5, 6]])
        >>> conversion_pred(X)
        array([  3.5,  19.5])

    """
    return (
        81.09
        + 1.0284 * X[:, 0]
        + 4.043 * X[:, 1]
        + 6.2037 * X[:, 2]
        - 1.8366 * X[:, 0] ** 2
        + 2.9382 * X[:, 1] ** 2
        - 5.1915 * X[:, 2] ** 2
        + 2.2150 * X[:, 0] * X[:, 1]
        + 11.375 * X[:, 0] * X[:, 2]
        - 3.875 * X[:, 1] * X[:, 2]
    )


def activity_pred(X) -> np.ndarray:
    """
    Compute activity predictions for each row in the input array.

    Args:
        X (np.ndarray): 2D array where each row is a configuration.

    Returns:
        np.ndarray: 1D array of activity predictions.

    Examples:
        >>> import numpy as np
        >>> from spotoptim.function.mo import activity_pred
        >>> # Example input data
        >>> X = np.array([[1, 2, 3], [4, 5, 6]])
        >>> activity_pred(X)
        array([  1.5,  10.5])
    """
    return (
        59.85
        + 3.583 * X[:, 0]
        + 0.2546 * X[:, 1]
        + 2.2298 * X[:, 2]
        + 0.83479 * X[:, 0] ** 2
        + 0.07484 * X[:, 1] ** 2
        + 0.05716 * X[:, 2] ** 2
        - 0.3875 * X[:, 0] * X[:, 1]
        - 0.375 * X[:, 0] * X[:, 2]
        + 0.3125 * X[:, 1] * X[:, 2]
    )


def fun_myer16a(X, fun_control=None) -> np.ndarray:
    """
    Compute both conversion and activity predictions for each row in the input array.

    Notes:
        Implements a response surface experiment described by Myers, Montgomery, and Anderson-Cook (2016).
        The function computes two objectives: conversion and activity.

    References:
        - Myers, R. H., Montgomery, D. C., and Anderson-Cook, C. M. Response surface methodology:
          process and product optimization using designed experiments. John Wiley & Sons, 2016.
        - Kuhn, M. desirability: Function optimization and ranking via desirability functions. Tech. rep., 9 2016.

    Args:
        X (np.ndarray): 2D array where each row is a configuration.
        fun_control (dict, optional): Additional control parameters (not used here).

    Returns:
        np.ndarray: 2D array where each row contains [conversion_pred, activity_pred].

    Examples:
        >>> import numpy as np
        >>> from spotoptim.function.mo import fun_myer16a
        >>> # Example input data
        >>> X = np.array([[1, 2, 3], [4, 5, 6]])
        >>> fun_myer16a(X)
        array([[  3.5,   1.5],
               [ 19.5,  10.5]])
    """
    return np.column_stack((conversion_pred(X), activity_pred(X)))
