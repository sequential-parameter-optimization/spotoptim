import numpy as np
from typing import List, Tuple, Union, Optional
from numpy.random import Generator, default_rng
from sklearn.datasets import make_blobs


def generate_uniform_design(
    bounds: Union[List[Tuple[float, float]], np.ndarray],
    n_design: int,
    seed: Optional[Union[int, Generator]] = None,
) -> np.ndarray:
    """Generate a uniform random experimental design.

    Generates n_design points uniformly distributed within the specified bounds.
    This function is compatible with SpotOptim's random number handling.

    Args:
        bounds (Union[List[Tuple[float, float]], np.ndarray]): Design space bounds.
            List of (lower, upper) tuples for each dimension.
        n_design (int): Number of design points to generate.
        seed (Optional[Union[int, Generator]], optional): Random seed or generator.
            Defaults to None.

    Returns:
        np.ndarray: Generated design points of shape (n_design, n_dim).

    Examples:
        >>> import numpy as np
        >>> from spotoptim.sampling.design import generate_uniform_design
        >>> bounds = [(-5, 5), (0, 10)]
        >>> X = generate_uniform_design(bounds, n_design=5, seed=42)
        >>> X.shape
        (5, 2)
        >>> np.all((X >= [-5, 0]) & (X <= [5, 10]))
        True
    """
    # Initialize random number generator
    rng = default_rng(seed)

    # Convert bounds to numpy array for easier handling
    bounds_arr = np.array(bounds)
    n_dim = len(bounds_arr)

    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]

    # Generate random points in [0, 1] range
    X_unit = rng.random(size=(n_design, n_dim))

    # Scale to bounds
    X = lower + X_unit * (upper - lower)

    return X


def generate_collinear_design(
    bounds: Union[List[Tuple[float, float]], np.ndarray],
    n_design: int,
    sigma: float = 0.01,
    seed: Optional[Union[int, Generator]] = None,
) -> np.ndarray:
    """Generates a collinear design (poorly projected).

    Currently implemented for 2D designs only. Generates points along a line
    with some Gaussian noise. The points are scaled to the provided bounds.

    Args:
        bounds (Union[List[Tuple[float, float]], np.ndarray]): Design space bounds.
        n_design (int): The number of points to generate.
        sigma (float): The standard deviation of the noise added to the y-coordinate
            (relative to unit scale). Defaults to 0.01.
        seed (Optional[Union[int, Generator]], optional): Random seed or generator.

    Returns:
        np.ndarray: A 2D array of shape (n_design, n_dim) with collinear points.

    Raises:
        ValueError: If dimension is not 2.

    Examples:
        >>> import numpy as np
        >>> from spotoptim.sampling.design import generate_collinear_design
        >>> bounds = [(0, 1), (0, 1)]
        >>> X = generate_collinear_design(bounds, n_design=10, seed=42)
        >>> X.shape
        (10, 2)
    """
    # Initialize random number generator
    rng = default_rng(seed)

    bounds_arr = np.array(bounds)
    n_dim = len(bounds_arr)

    if n_dim != 2:
        raise ValueError("Collinear design currently implemented for 2D only.")

    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]

    # Generate points in [0, 1] range first
    x_coords = np.linspace(0.1, 0.9, n_design)

    # Linear relationship: y = 0.2*x + 0.3 + noise
    # This keeps points roughly in [0.3, 0.5] range for y (before noise)
    y_coords = 0.2 * x_coords + 0.3

    # Add noise
    noise = rng.normal(0, sigma, n_design)
    y_coords = y_coords + noise

    # Clip to [0, 1] to be safe
    y_coords = np.clip(y_coords, 0.0, 1.0)

    # Stack unit coordinates
    X_unit = np.vstack([x_coords, y_coords]).T

    # Scale to bounds
    X = lower + X_unit * (upper - lower)

    return X


def generate_clustered_design(
    bounds: Union[List[Tuple[float, float]], np.ndarray],
    n_design: int,
    n_clusters: int,
    seed: Optional[Union[int, Generator]] = None,
) -> np.ndarray:
    """Generates a clustered design.

    Generates clusters of points using sklearn.datasets.make_blobs.
    Points are scaled to the provided bounds.

    Args:
        bounds (Union[List[Tuple[float, float]], np.ndarray]): Design space bounds.
        n_design (int): The number of points to generate.
        n_clusters (int): The number of clusters.
        seed (Optional[Union[int, Generator]], optional): Random seed or generator.

    Returns:
        np.ndarray: A 2D array of shape (n_design, n_dim) with clustered points.
    """
    bounds_arr = np.array(bounds)
    n_dim = len(bounds_arr)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]

    # Handle seed for make_blobs (expects int or RandomState or None)
    random_state = seed
    if isinstance(seed, np.random.Generator):
        # Generator doesn't map directly to RandomState, but we can get integers
        random_state = seed.integers(0, 2**32 - 1)

    X_unit, _ = make_blobs(
        n_samples=n_design,
        n_features=n_dim,
        centers=n_clusters,
        cluster_std=0.05,
        random_state=random_state,
        center_box=(0.1, 0.9),
    )

    # Normalize to [0, 1] if values exceed bounds
    X_min = X_unit.min(axis=0)
    X_max = X_unit.max(axis=0)

    if np.any(X_min < 0) or np.any(X_max > 1):
        X_unit = (X_unit - X_min) / (X_max - X_min + 1e-6)

    # Scale to bounds
    X = lower + X_unit * (upper - lower)

    return X
