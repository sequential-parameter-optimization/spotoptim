import numpy as np


def normalize_X(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize array X to [0, 1] in each dimension.

    For dimensions where all values are identical (X_max == X_min), the normalized
    value is set to 0.5 to avoid division by zero.

    Args:
        X (np.ndarray): Input array of shape (n, d) to normalize.
        eps (float, optional): Small value to avoid division by zero when range is very small.
            Defaults to 1e-12.

    Returns:
        np.ndarray: Normalized array with values in [0, 1] for each dimension.
            For constant dimensions, values are set to 0.5.

    Examples:
        >>> from spotoptim.utils.stats import normalize_X
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> normalize_X(X)
        array([[0. , 0. ],
               [0.5, 0.5],
               [1. , 1. ]])

        >>> # Constant dimension example
        >>> X_const = np.array([[1, 5], [1, 5], [1, 5]])
        >>> normalize_X(X_const)
        array([[0.5, 0.5],
               [0.5, 0.5],
               [0.5, 0.5]])
    """
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_range = X_max - X_min

    # Handle constant dimensions (where range is effectively zero)
    constant_dims = X_range < eps

    # Normalize non-constant dimensions
    X_normalized = np.zeros_like(X, dtype=float)
    X_normalized[:, ~constant_dims] = (X[:, ~constant_dims] - X_min[~constant_dims]) / (
        X_range[~constant_dims]
    )

    # Set constant dimensions to 0.5
    X_normalized[:, constant_dims] = 0.5

    return X_normalized
