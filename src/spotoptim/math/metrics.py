import numpy as np
from sklearn import metrics
from sklearn.metrics import pairwise_distances


def magnitude(x, y) -> float:
    """
    Calculate the squared magnitude of the hypoteneuse of two arrays.
    Identical to the sum of squared elements of two arrays.

    Args:
        x (numpy.ndarray): The first array.
        y (numpy.ndarray): The second array.

    Returns:
        float: The squared magnitude of the hypoteneuse of the two arrays.

    Examples:
        >>> from spotoptim.math.metrics import magnitude
        >>> import numpy as np
        >>> x = np.array([1., 2., 3.])
        >>> y = np.array([4., 5., 6.])
        >>> magnitude(x, y)
        91.0
        # Calculation: np.hypot(1, 4)^2 + np.hypot(2, 5)^2 + np.hypot(3, 6)^2
        # = (sqrt(1^2 + 4^2))^2 + (sqrt(2^2 + 5^2))^2 + (sqrt(3^2 + 6^2))^2
        # = (sqrt(17))^2 + (sqrt(29))^2 + (sqrt(45))^2 = 17 + 29 + 45 = 91
    """
    return np.sum(np.hypot(x, y) ** 2)


def compute_using_metric_function(p1_x, p1_y, p2_x, p2_y, metric_function, **kwargs) -> float:
    """
    Computes the metric using a specified sklearn metric function.

    Args:
        p1_x (np.ndarray): Array for the x-coordinates of the first point.
        p1_y (np.ndarray): Array for the y-coordinates of the first point.
        p2_x (np.ndarray): Array for the x-coordinates of the second point.
        p2_y (np.ndarray): Array for the y-coordinates of the second point.
        metric_function (callable): The metric function from sklearn.
        **kwargs (any): Additional arguments for the metric function.

    Returns:
        float: The computed metric value.

    Examples:
        >>> from sklearn.metrics import mean_squared_error
            from spotoptim.math.metrics import compute_using_metric_function
            p1_x = [1.0]
            p1_y = [2.0]
            p2_x = [3.0]
            p2_y = [4.0]
            compute_using_metric_function(p1_x, p1_y, p2_x, p2_y, mean_squared_error)
                4.0
        >>> from sklearn.metrics import mean_absolute_error
            from spotoptim.math.metrics import compute_using_metric_function
            p1_x = [0.0]
            p1_y = [0.0]
            p2_x = [2.0]
            p2_y = [3.0]
            compute_using_metric_function(p1_x, p1_y, p2_x, p2_y, mean_absolute_error)
                2.5
    """
    p1 = np.array([p1_x, p1_y])
    p2 = np.array([p2_x, p2_y])
    result = metric_function(p1, p2, **kwargs)
    return float(result)


def compute_pairwise_distance(p1_x, p1_y, p2_x, p2_y, metric_name, **kwargs) -> float:
    """
    Computes the pairwise distance using sklearn's pairwise_distances.
    Distance is calculated between two 2-dim points P1 = (p1_x, p1_y) and
    P2 = (p2_x, p2_y) using a specified distance from sklearn, such as
        * "seuclidean"
        * "chebyshev"
        * "cityblock"
        * "dice"
        * "precomputed"
        * "l2"
        * "correlation"
        * "sokalsneath"
        * "rogerstanimoto"
        * "sqeuclidean"
        * "russellrao"
        * "mahalanobis"
        * "braycurtis"
        * "manhattan"
        * "nan_euclidean"
        * "yule"
        * "euclidean"
        * "hamming"
        * "haversine"
        * "l1"
        * "minkowski"
        * "cosine"
        * "sokalmichener"
        * "wminkowski"
        * "canberra"
        * "jaccard"
        * "matching"

    Args:
        p1_x (np.ndarray): Array for the x-coordinates of the first point.
        p1_y (np.ndarray): Array for the y-coordinates of the first point.
        p2_x (np.ndarray): Array for the x-coordinates of the second point.
        p2_y (np.ndarray): Array for the y-coordinates of the second point.
        metric_name (str): The name of the metric.
        **kwargs (any): Additional arguments for the metric function.

    Returns:
        float: The computed distance value.

    Examples:
        >>> from spotoptim.math.metrics import compute_pairwise_distance
            p1_x = [0.0]
            p1_y = [0.0]
            p2_x = [3.0]
            p2_y = [4.0]
            compute_pairwise_distance(p1_x, p1_y, p2_x, p2_y, metric_name="euclidean")
                5.0
        >>> from spotoptim.math.metrics import compute_pairwise_distance
            p1_x = [0.0]
            p1_y = [0.0]
            p2_x = [1.0]
            p2_y = [1.0]
            compute_pairwise_distance(p1_x, p1_y, p2_x, p2_y, metric_name="euclidean")
                1.4142135623730951
    """
    p1 = np.column_stack((p1_x, p1_y))
    p2 = np.column_stack((p2_x, p2_y))
    distances = pairwise_distances(p1, p2, metric=metric_name, **kwargs)
    return float(distances[0, 0])


def sklearn_metric(p1_x, p1_y, p2_x, p2_y, metric_name="mean_squared_error", **kwargs) -> float:
    """
    Calculate a metric between two 2-dim points P1 = (p1_x, p1_y) and
    P2 = (p2_x, p2_y) using a specified sklearn metric.

    Args:
        p1_x (float): The x-coordinate of the first point.
        p1_y (float): The y-coordinate of the first point.
        p2_x (float): The x-coordinate of the second point.
        p2_y (float): The y-coordinate of the second point.
        metric_name (str): The name of the sklearn metric function to use.
        **kwargs (any): Additional keyword arguments to pass to the metric function.

    Returns:
        float: The result of the metric calculation.

    Raises:
        ValueError: If the specified metric_name is not found in sklearn.metrics.

    Examples:
        >>> from spotoptim.math.metrics import sklearn_metric
        >>> p1_x = 0.0
        >>> p1_y = 0.0
        >>> p2_x = 1.0
        >>> p2_y = 1.0
        >>> sklearn_metric(p1_x, p1_y, p2_x, p2_y, metric_name="euclidean_distances")
        1.41421356
        >>> sklearn_metric(p1_x, p1_y, p2_x, p2_y, metric_name="mean_squared_error")
        1.0
    """
    pairwise_distance = {
        "seuclidean",
        "chebyshev",
        "cityblock",
        "dice",
        "precomputed",
        "l2",
        "correlation",
        "sokalsneath",
        "rogerstanimoto",
        "sqeuclidean",
        "russellrao",
        "mahalanobis",
        "braycurtis",
        "manhattan",
        "nan_euclidean",
        "yule",
        "euclidean",
        "hamming",
        "haversine",
        "l1",
        "minkowski",
        "cosine",
        "sokalmichener",
        "wminkowski",
        "canberra",
        "jaccard",
        "matching",
    }

    # Ensure inputs are numpy arrays, even if they were single floats initially
    if isinstance(p1_x, (float, int)):
        p1_x = np.array([p1_x])
        p1_y = np.array([p1_y])
        p2_x = np.array([p2_x])
        p2_y = np.array([p2_y])

    try:
        metric_function = getattr(metrics, metric_name, None)

        if metric_name in pairwise_distance:
            return compute_pairwise_distance(p1_x, p1_y, p2_x, p2_y, metric_name, **kwargs)

        if metric_function:
            return compute_using_metric_function(p1_x, p1_y, p2_x, p2_y, metric_function, **kwargs)

        raise ValueError(f"Metric '{metric_name}' not found in sklearn.metrics")
    except TypeError as e:
        raise ValueError(f"Error computing the metric '{metric_name}': {e}")
    except Exception as e:
        raise ValueError(f"An error occurred: {e}")
