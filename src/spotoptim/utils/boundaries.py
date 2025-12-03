import numpy as np
import pandas as pd
from typing import Union


def get_boundaries(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the minimum and maximum values for each column in a NumPy array.

    Args:
        data (np.ndarray): A NumPy array of shape (n, k), where n is the number of rows
            and k is the number of columns.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
            - The first array contains the minimum values for each column, with shape (k,).
            - The second array contains the maximum values for each column, with shape (k,).

    Raises:
        ValueError: If the input array has shape (1, 0) (empty array).

    Examples:
        >>> from spotoptim.utils.boundaries import get_boundaries
        >>> import numpy as np
        >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> min_values, max_values = get_boundaries(data)
        >>> print("Minimum values:", min_values)
        Minimum values: [1 2 3]
        >>> print("Maximum values:", max_values)
        Maximum values: [7 8 9]
    """
    if data.size == 0:
        raise ValueError("Input array cannot be empty.")
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)
    return min_values, max_values


def map_to_original_scale(
    X_search: Union[pd.DataFrame, np.ndarray],
    x_min: np.ndarray,
    x_max: np.ndarray,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Maps the values in X_search from the range [0, 1] to the original scale defined by x_min and x_max.

    Args:
        X_search (Union[pd.DataFrame, np.ndarray]): A Pandas DataFrame or NumPy array containing the search points in the range [0, 1].
        x_min (np.ndarray): A NumPy array containing the minimum values for each feature in the original scale.
        x_max (np.ndarray): A NumPy array containing the maximum values for each feature in the original scale.

    Returns:
        Union[pd.DataFrame, np.ndarray]: A Pandas DataFrame or NumPy array with the values mapped to the original scale.

    Examples:
        >>> from spotoptim.utils.boundaries import map_to_original_scale
        >>> import numpy as np
        >>> import pandas as pd
        >>> X_search = pd.DataFrame([[0.5, 0.5], [0.25, 0.75]], columns=['x', 'y'])
        >>> x_min = np.array([0, 0])
        >>> x_max = np.array([10, 20])
        >>> X_search_scaled = map_to_original_scale(X_search, x_min, x_max)
        >>> print(X_search_scaled)
              x     y
        0   5.0  10.0
        1   2.5  15.0
    """
    if not isinstance(X_search, (pd.DataFrame, np.ndarray)):
        raise TypeError("X_search must be a Pandas DataFrame or a NumPy array.")

    # if x_min or x_max are not numpy arrays, convert them to numpy arrays
    if not isinstance(x_min, np.ndarray):
        x_min = np.array(x_min)
    if not isinstance(x_max, np.ndarray):
        x_max = np.array(x_max)

    if len(x_min) != X_search.shape[1]:
        raise IndexError(
            f"x_min and X_search must have the same number of columns. x_min has {len(x_min)} columns and X_search has {X_search.shape[1]} columns."
        )
    if len(x_max) != X_search.shape[1]:
        raise IndexError(
            f"x_max and X_search must have the same number of columns. x_max has {len(x_max)} columns and X_search has {X_search.shape[1]} columns."
        )

    if isinstance(X_search, pd.DataFrame):
        X_search_scaled = (
            X_search.copy()
        )  # Create a copy to avoid modifying the original DataFrame
        for i, col in enumerate(X_search.columns):
            X_search_scaled.loc[:, col] = (
                X_search[col] * (x_max[i] - x_min[i]) + x_min[i]
            )
        return X_search_scaled
    elif isinstance(X_search, np.ndarray):
        X_search_scaled = (
            X_search.copy()
        )  # Create a copy to avoid modifying the original array
        for i in range(X_search.shape[1]):
            X_search_scaled[:, i] = X_search[:, i] * (x_max[i] - x_min[i]) + x_min[i]
        return X_search_scaled
