import numpy as np


from typing import Optional, Union
from numpy.random import Generator, default_rng


def rlh(
    n: int, k: int, edges: int = 0, seed: Optional[Union[int, Generator]] = None
) -> np.ndarray:
    """
    Generates a random Latin hypercube within the [0,1]^k hypercube.

    Args:
        n (int): Desired number of points.
        k (int): Number of design variables (dimensions).
        edges (int, optional):
            If 1, places centers of the extreme bins at the domain edges ([0,1]).
            Otherwise, bins are fully contained within the domain, i.e. midpoints.
            Defaults to 0.

    Returns:
        np.ndarray: A Latin hypercube sampling plan of n points in k dimensions,
                    with each coordinate in the range [0,1].

    Examples:
        >>> from spotoptim.sampling.lhs import rlh
        >>> import numpy as np
        >>> # Generate a 2D Latin hypercube with 5 points and edges=0
        >>> X = rlh(n=5, k=2, edges=0)
        >>> print(X)
        # Example output (values vary due to randomness):
        # [[0.1  0.5 ]
        #  [0.7  0.1 ]
        #  [0.9  0.7 ]
        #  [0.3  0.9 ]
        #  [0.5  0.3 ]]
    """
    # Validate inputs
    if n < 1:
        raise ValueError("n must be >= 1")
    if k < 1:
        raise ValueError("k must be >= 1")
    if edges not in (0, 1):
        raise ValueError("edges must be 0 or 1")

    # Initialize array
    X = np.zeros((n, k), dtype=float)

    # Initialize rng
    rng = default_rng(seed)

    # Fill with random permutations
    for i in range(k):
        X[:, i] = rng.permutation(n)

    # Adjust normalization based on the edges flag
    if edges == 1:
        # [X=0..n-1] -> [0..1]
        if n == 1:
            # Avoid division by zero; for a single point place at 0
            X[:, :] = 0.0
        else:
            X = X / (n - 1)
    else:
        # Points at true midpoints
        # [X=0..n-1] -> [0.5/n..(n-0.5)/n]
        X = (X + 0.5) / n

    return X
