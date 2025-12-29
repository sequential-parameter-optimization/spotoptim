import numpy as np
from typing import List, Tuple, Union, Optional
from numpy.random import Generator, default_rng
from sklearn.datasets import make_blobs
from scipy.stats import qmc


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

    Examples:
        >>> import numpy as np
        >>> from spotoptim.sampling.design import generate_clustered_design
        >>> bounds = [(-5, 5), (0, 10)]
        >>> X = generate_clustered_design(bounds, n_design=5, n_clusters=2, seed=42)
        >>> X.shape
        (5, 2)
        >>> np.all((X >= [-5, 0]) & (X <= [5, 10]))
        True
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


def generate_sobol_design(
    bounds: Union[List[Tuple[float, float]], np.ndarray],
    n_design: int,
    seed: Optional[Union[int, Generator]] = None,
) -> np.ndarray:
    """Generates a Sobol sequence design.

    Args:
        bounds (Union[List[Tuple[float, float]], np.ndarray]): Design space bounds.
        n_design (int): The number of points to generate.
        seed (Optional[Union[int, Generator]], optional): Random seed or generator.

    Returns:
        np.ndarray: An array of shape (n_design, n_dim) containing the generated Sobol sequence points.

    Notes:
        - The Sobol sequence is generated with a length that is a power of 2.
        - Scrambling is enabled for improved uniformity.

    Examples:
        >>> import numpy as np
        >>> from spotoptim.sampling.design import generate_sobol_design
        >>> bounds = [(-5, 5), (0, 10)]
        >>> X = generate_sobol_design(bounds, n_design=5, seed=42)
        >>> X.shape
        (5, 2)
    """
    bounds_arr = np.array(bounds)
    n_dim = len(bounds_arr)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]

    # Handle seed
    random_state = seed
    if isinstance(seed, np.random.Generator):
        random_state = seed.integers(0, 2**32 - 1)

    sampler = qmc.Sobol(d=n_dim, scramble=True, seed=random_state)
    m = int(np.ceil(np.log2(n_design)))
    X_unit = sampler.random_base2(m=m)[:n_design, :]

    # Scale to bounds
    X = lower + X_unit * (upper - lower)

    return X


def generate_qmc_lhs_design(
    bounds: Union[List[Tuple[float, float]], np.ndarray],
    n_design: int,
    seed: Optional[Union[int, Generator]] = None,
) -> np.ndarray:
    """Generates a Latin Hypercube Sampling design using QMC.

    Args:
        bounds (Union[List[Tuple[float, float]], np.ndarray]): Design space bounds.
        n_design (int): The number of points to generate.
        seed (Optional[Union[int, Generator]], optional): Random seed or generator.

    Returns:
        np.ndarray: An array of shape (n_design, n_dim) containing the generated LHS points.

    Examples:
        >>> import numpy as np
        >>> from spotoptim.sampling.design import generate_qmc_lhs_design
        >>> bounds = [(-5, 5), (0, 10)]
        >>> X = generate_qmc_lhs_design(bounds, n_design=5, seed=42)
        >>> X.shape
        (5, 2)
    """
    bounds_arr = np.array(bounds)
    n_dim = len(bounds_arr)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]

    # Handle seed
    random_state = seed
    if isinstance(seed, np.random.Generator):
        random_state = seed.integers(0, 2**32 - 1)

    sampler = qmc.LatinHypercube(d=n_dim, seed=random_state)
    X_unit = sampler.random(n=n_design)

    # Scale to bounds
    X = lower + X_unit * (upper - lower)

    return X


def generate_grid_design(
    bounds: Union[List[Tuple[float, float]], np.ndarray],
    n_design: int,
    seed: Optional[Union[int, Generator]] = None,
) -> np.ndarray:
    """Generates a regular grid design.

    Points are generated by creating a regular grid where the number of points
    per dimension is derived from n_design (floor(n_design^(1/n_dim))).

    Note: The actual number of points returned might be less than n_design
    if n_design is not a perfect power of n_dim.

    Args:
        bounds (Union[List[Tuple[float, float]], np.ndarray]): Design space bounds.
        n_design (int): The target number of points. Used to determine points per dimension.
        seed (Optional[Union[int, Generator]], optional): Unused, kept for API consistency.

    Returns:
        np.ndarray: A 2D array of shape (points_per_dim^n_dim, n_dim) with grid points.

    Examples:
        >>> import numpy as np
        >>> from spotoptim.sampling.design import generate_grid_design
        >>> bounds = [(0, 1), (0, 1)]
        >>> X = generate_grid_design(bounds, n_design=25) # 5^2 = 25
        >>> X.shape
        (25, 2)
    """
    bounds_arr = np.array(bounds)
    n_dim = len(bounds_arr)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]

    if n_dim != 2:
        # Check if we want to support > 2D. The original code raised error.
        # But meshgrid supports N-D. Let's try to support it or keep restriction.
        # Original code: if self.k != 2: raise ValueError...
        # Let's keep restriction for now as specifically requested to fix, but maybe lift it?
        # User said "improve the code". Supporting N-D is an improvement.
        pass

    # Calculate points per dimension
    points_per_dim = int(np.floor(n_design ** (1 / n_dim)))
    if points_per_dim < 2:
        points_per_dim = 2  # Minimum 2 points per dim to have a grid?

    # Create grid in [0, 1]
    ticks = np.linspace(0, 1, points_per_dim, endpoint=True)

    if n_dim == 2:
        x, y = np.meshgrid(ticks, ticks)
        X_unit = np.vstack([x.ravel(), y.ravel()]).T
    else:
        # General N-D grid
        grid = np.meshgrid(*([ticks] * n_dim))
        X_unit = np.vstack([g.ravel() for g in grid]).T

    # Scale to bounds
    X = lower + X_unit * (upper - lower)

    return X


def fullfactorial(q, Edges=1) -> np.ndarray:
    """Generates a full factorial sampling plan in the unit cube.

    Args:
        q (list or np.ndarray):
            A list or array containing the number of points along each dimension (k-vector).
        Edges (int, optional):
            Determines spacing of points. If `Edges=1`, points are equally spaced from edge to edge (default).
            Otherwise, points will be in the centers of n = q[0]*q[1]*...*q[k-1] bins filling the unit cube.

    Returns:
        (np.ndarray): Full factorial sampling plan as an array of shape (n, k), where n is the total number of points and k is the number of dimensions.

    Raises:
        ValueError: If any dimension in `q` is less than 2.

    Notes:
        Many thanks to the original author of this code, A Sobester, for providing the original Matlab code under the GNU Licence.
        Original Matlab Code: Copyright 2007 A Sobester:
        "This program is free software: you can redistribute it and/or modify  it
        under the terms of the GNU Lesser General Public License as published by
        the Free Software Foundation, either version 3 of the License, or any
        later version.
        This program is distributed in the hope that it will be useful, but
        WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
        General Public License for more details.
        You should have received a copy of the GNU General Public License and GNU
        Lesser General Public License along with this program. If not, see
        <http://www.gnu.org/licenses/>."

    Examples:
        >>> from spotpython.utils.sampling import fullfactorial
        >>> q = [3, 2]
        >>> X = fullfactorial(q, Edges=0)
        >>> print(X)
                [[0.         0.        ]
                [0.         0.75      ]
                [0.41666667 0.        ]
                [0.41666667 0.75      ]
                [0.83333333 0.        ]
                [0.83333333 0.75      ]]
        >>> X = fullfactorial(q, Edges=1)
        >>> print(X)
                [[0.  0. ]
                [0.  1. ]
                [0.5 0. ]
                [0.5 1. ]
                [1.  0. ]
                [1.  1. ]]

    """
    q = np.array(q)
    if np.min(q) < 2:
        raise ValueError("You must have at least two points per dimension.")

    # Total number of points in the sampling plan
    n = np.prod(q)

    # Number of dimensions
    k = len(q)

    # Pre-allocate memory for the sampling plan
    X = np.zeros((n, k))

    # Additional phantom element
    q = np.append(q, 1)

    for j in range(k):
        if Edges == 1:
            one_d_slice = np.linspace(0, 1, q[j])
        else:
            one_d_slice = np.linspace(1 / (2 * q[j]), 1, q[j]) - 1 / (2 * q[j])

        column = np.array([])

        while len(column) < n:
            for ll in range(q[j]):
                column = np.append(
                    column, np.ones(np.prod(q[j + 1 : k])) * one_d_slice[ll]
                )

        X[:, j] = column

    return X
