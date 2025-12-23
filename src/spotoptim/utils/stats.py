import numpy as np
import pandas as pd
from typing import Union
from scipy.stats import norm


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
    # Handle empty array
    if X.size == 0:
        return X.copy()

    # Ensure X is a numpy array (handles DataFrames)
    X = np.asarray(X)

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


def calculate_outliers(
    series_or_df: Union[pd.Series, pd.DataFrame], irqmultiplier: float = 1.5
) -> int:
    """
    Calculate the number of outliers using the IQR method.

    Accepts either a pandas Series or a pandas DataFrame. For a DataFrame,
    counts outliers across all numeric columns and returns the total count.

    Args:
        series_or_df: pd.Series or pd.DataFrame containing numeric data.
        irqmultiplier (float, optional): Multiplier for IQR to define fences. Defaults to 1.5.

    Returns:
        int: The number of outliers.

    Examples:
        >>> import pandas as pd
        >>> from spotoptim.utils.stats import calculate_outliers
        >>> s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100])
        >>> calculate_outliers(s)
        1

        >>> df = pd.DataFrame({
        ...     'a': [1, 2, 3, 100],
        ...     'b': [10, 12, 11, 10]
        ... })
        >>> calculate_outliers(df)
        1
    """
    if isinstance(series_or_df, pd.Series):
        s = series_or_df.dropna()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - irqmultiplier * iqr
        upper_fence = q3 + irqmultiplier * iqr
        return int(s[(s < lower_fence) | (s > upper_fence)].shape[0])

    if isinstance(series_or_df, pd.DataFrame):
        df = series_or_df.select_dtypes(include=["number"]).dropna(how="all")
        total_outliers = 0
        for col in df.columns:
            s = df[col].dropna()
            if s.empty:
                continue
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower_fence = q1 - irqmultiplier * iqr
            upper_fence = q3 + irqmultiplier * iqr
            total_outliers += int(s[(s < lower_fence) | (s > upper_fence)].shape[0])
        return total_outliers

    raise TypeError("Input must be a pandas Series or DataFrame")


def get_combinations(ind_list: list, type="indices") -> list:
    """
    Generates all possible combinations of two values from a list of values. Order is not important.

    Args:
        ind_list (list): A list of target indices.
        type (str): The type of output, either 'values' or 'indices'. Default is 'indices'.

    Returns:
        list: A list of tuples, where each tuple contains a combination of two values.
            The order of the values within a tuple is not important, and each combination appears only once.

    Examples:
        >>> from spotoptim.utils import get_combinations
        >>> ind_list = [0, 10, 20, 30]
        >>> combinations = get_combinations(ind_list)
        >>> combinations = get_combinations(ind_list, type='indices')
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        >>> print(combinations, type='values')
            [(0, 10), (0, 20), (0, 30), (1, 20), (1, 30), (2, 30)]
    """
    # check that ind_list is a list
    if not isinstance(ind_list, list):
        raise ValueError("ind_list must be a list.")
    m = len(ind_list)
    if type == "values":
        combinations = [
            (ind_list[i], ind_list[j]) for i in range(m) for j in range(i + 1, m)
        ]
    elif type == "indices":
        combinations = [(i, j) for i in range(m) for j in range(i + 1, m)]
    else:
        raise ValueError("type must be either 'values' or 'indices'.")
    return combinations


def get_sample_size(alpha: float, beta: float, sigma: float, delta: float) -> float:
    """
    Calculate sample size n for comparing two means.

    Formula: n = 2 * sigma^2 * (z_{1-alpha/2} + z_{1-beta})^2 / delta^2
    This corresponds to a two-sided test with equal variance.

    Args:
        alpha (float): Significance level (Type I error probability).
        beta (float): Type II error probability (1 - Power).
        sigma (float): Standard deviation of the population (assumed equal for both groups).
        delta (float): Minimum detectable difference (effect size to detect).

    Returns:
        float: The required sample size n per group.

    Examples:
        >>> from spotoptim.utils.stats import get_sample_size
        >>> alpha = 0.05
        >>> beta = 0.2  # Power = 80%
        >>> sigma = 1.0
        >>> delta = 1.0
        >>> n = get_sample_size(alpha, beta, sigma, delta)
        >>> print(f"{n:.4f}")
        15.6978
    """
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(1 - beta)
    n = 2 * sigma**2 * (z_alpha + z_beta) ** 2 / delta**2
    return n
