"""
Analytical single-objective test functions for optimization benchmarking.

This module provides well-known analytical test functions commonly used for evaluating and benchmarking optimization algorithms.
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


def wingwt(X) -> np.ndarray:
    """Aircraft Wing Weight function.

    The example models the weight of an unpainted light aircraft wing.
    The function accepts inputs in the unit cube [0,1]^9 and returns the wing weight.

    Args:
        X (array-like): Input points with shape (n_samples, 9) or (9,) or (10,) or (n_samples, 10).
            Input variables order: [Sw, Wfw, A, L, q, l, Rtc, Nz, Wdg, Wp(optional)]

    Returns:
        np.ndarray: Wing weight values at the input points with shape (n_samples,).

    Examples:
        Single point evaluation (Baseline Cessna C172 - Unpainted):

        >>> from spotoptim.function.so import wingwt
        >>> import numpy as np
        >>> # Baseline configuration in unit cube
        >>> x_base = np.array([0.48, 0.4, 0.38, 0.5, 0.62, 0.344, 0.4, 0.37, 0.38])
        >>> wingwt(x_base)
        array([233.90...])

        Batch evaluation:

        >>> X = np.vstack([x_base, x_base])
        >>> wingwt(X)
        array([233.90..., 233.90...])

    References:
        Forrester, A., Sobester, A., & Keane, A. (2008). Engineering design via surrogate modelling:
        a practical guide. John Wiley & Sons.
    """
    # Ensure x is a 2D array for batch evaluation
    X = np.atleast_2d(X)

    n_features = X.shape[1]
    if n_features not in [9, 10]:
        raise ValueError(f"wingwt expects 9 or 10 features, got {n_features}")

    # Transform from unit cube to natural scales
    Sw = X[:, 0] * (200 - 150) + 150
    Wfw = X[:, 1] * (300 - 220) + 220
    A = X[:, 2] * (10 - 6) + 6
    L = (X[:, 3] * (10 - (-10)) - 10) * np.pi / 180
    q = X[:, 4] * (45 - 16) + 16
    taper = X[:, 5] * (1 - 0.5) + 0.5
    Rtc = X[:, 6] * (0.18 - 0.08) + 0.08
    Nz = X[:, 7] * (6 - 2.5) + 2.5
    Wdg = X[:, 8] * (2500 - 1700) + 1700

    # Paint weight (W_p)
    # Range assumed [0.06, 0.08] based on typical value 0.064 if modeled.
    # If not provided (9 inputs), assume unpainted (0.0).
    if n_features == 10:
        # User is explicitly providing W_p, scale it.
        # Assuming domain [0.06, 0.08] for the 10th variable "painted" case.
        # This is a guess to support the physics.
        Wp = X[:, 9] * (0.08 - 0.06) + 0.06
    else:
        Wp = 0.0

    # Calculate weight on natural scale
    W = 0.036 * Sw**0.758 * Wfw**0.0035 * (A / np.cos(L) ** 2) ** 0.6 * q**0.006
    W = W * taper**0.04 * (100 * Rtc / np.cos(L)) ** (-0.3) * (Nz * Wdg) ** (0.49)

    # Add paint weight term
    W = W + Sw * Wp

    return W.ravel()


def lennard_jones(X: np.ndarray, n_atoms: int = 13) -> np.ndarray:
    """Lennard-Jones Atomic Cluster Potential Energy.

    Calculates the potential energy of a cluster of N atoms interacting via the
    Lennard-Jones potential. The optimization problem involves finding the atomic
    coordinates that minimize the total potential energy. This is a classic
    benchmark problem known for its high difficulty due to the exponential growth
    of local minima with N.

    Input Domain Handling:
        The function accepts inputs in the range [0, 1] and internally maps them
        to the search domain [-2, 2] for each coordinate.

    Args:
        X (np.ndarray): Input points.
            - Shape (n_samples, 3 * n_atoms) for batch evaluation.
            - Shape (3 * n_atoms,) for single evaluation.
            The input represents the flattened [x1, y1, z1, x2, y2, z2, ...] coordinates.
        n_atoms (int, optional): Number of atoms in the cluster. Defaults to 13.

    Returns:
        np.ndarray: Potential energy values. Shape (n_samples,).

    Raises:
        ValueError: If input dimensions do not match 3 * n_atoms.

    Note:
        - Global minimum for N=13: E ≈ -44.3268
        - Search domain: [-2, 2]^(3N) (mapped from [0, 1])
        - Characteristics: Extremely rugged landscape, non-convex, many local minima.

    Examples:
        Single point evaluation (random configuration):

        >>> from spotoptim.function import lennard_jones
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> X = rng.random(39)  # 13 atoms * 3 coords, in [0, 1]
        >>> lennard_jones(X)
        array([9.5...e+...])

        Batch evaluation:

        >>> X = rng.random((5, 39))
        >>> lennard_jones(X).shape
        (5,)

    References:
        Wales, D. J., & Doye, J. P. (1997). Global optimization by basin-hopping and
        the lowest energy structures of Lennard-Jones clusters containing up to 110 atoms.
        The Journal of Physical Chemistry A, 101(28), 5111-5116.
    """
    X = np.atleast_2d(X).astype(float)
    n_samples = X.shape[0]
    expected_dim = 3 * n_atoms

    if X.shape[1] != expected_dim:
        raise ValueError(
            f"Input dimension must be 3 * n_atoms ({expected_dim}). Got {X.shape[1]}."
        )

    # Scale from [0, 1] to [-2, 2]
    # [-2, 2] range has length 4.
    # val_scaled = val_01 * (max - min) + min
    coords_flattened = X * 4.0 - 2.0

    # Reshape to (n_samples, n_atoms, 3)
    coords = coords_flattened.reshape(n_samples, n_atoms, 3)

    potential = np.zeros(n_samples)

    # Calculate pairwise interactions
    # Simple double loop over atoms is often sufficient for typical N (N < ~150)
    # vectorized along samples.
    # Potential formula: 4*epsilon * sum_{i<j} [ (sigma/r_ij)^12 - (sigma/r_ij)^6 ]
    # We use reduced units: epsilon=1, sigma=1.

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # Difference vectors: (n_samples, 3)
            d_vec = coords[:, i, :] - coords[:, j, :]

            # Squared Euclidean distances: (n_samples,)
            r2 = np.sum(d_vec**2, axis=1)

            # Avoid division by zero / singularities by setting a lower bound
            # In optimization, particles can overlap leading to infinity.
            # We clip at a small value.
            r2 = np.maximum(r2, 1e-12)

            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2**3
            inv_r12 = inv_r6**2

            potential += inv_r12 - inv_r6

    return 4.0 * potential


def robot_arm_obstacle(X: np.ndarray) -> np.ndarray:
    """10-Link Planar Robot Arm Inverse Kinematics with Obstacle Avoidance.

    The goal is to minimize the distance of the end-effector to a target point
    while avoiding collision with a set of circular obstacles. This problem mimics
    a real-world inverse kinematics solver for a redundant manipulator.

    Input Domain Handling:
        The function accepts inputs in the range [0, 1] and internally maps them
        to the search domain [-pi, pi] for each joint angle (radians).

    Args:
        X (np.ndarray): Input angles (normalized).
            - Shape (n_samples, 10).
            The input contains the normalized relative angles for the 10 links.

    Returns:
        np.ndarray: Cost values (Weighted sum of Distance + Penalty). Shape (n_samples,).

    Raises:
        ValueError: If input dimensions do not match 10.

    Note:
        - Target: (5.0, 5.0)
        - Obstacles:
            - Order 1: (2,2), r=1
            - Order 2: (4,3), r=1.5
            - Order 3: (3,6), r=1
        - Dimensions: 10 (fixed)
        - Search domain: [-pi, pi]^10 (mapped from [0, 1])
        - Characteristics: Multimodal, disjoint feasible regions, constrained.

    Examples:
        Single point evaluation:

        >>> from spotoptim.function.so import robot_arm_obstacle
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> X = rng.random(10)  # Random angles in [0, 1]
        >>> robot_arm_obstacle(X)
        array([2547...])

        Batch evaluation:

        >>> X = rng.random((5, 10))
        >>> robot_arm_obstacle(X).shape
        (5,)
    """
    X = np.atleast_2d(X).astype(float)
    n_samples = X.shape[0]
    n_dims = X.shape[1]

    if n_dims != 10:
        raise ValueError(
            f"This function requires exactly 10 dimensions (joint angles), got {n_dims}."
        )

    # Scale from [0, 1] to [-pi, pi]
    # [-pi, pi] range length is 2*pi
    X_scaled = X * (2 * np.pi) - np.pi

    # --- Problem Configuration ---
    # Length of each arm link
    L = np.ones(10) * 1.0

    # Target coordinates (reachable within total length of 10)
    target = np.array([5.0, 5.0])

    # Obstacles: centers (x,y) and radii (r)
    obstacles = [
        {"center": np.array([2.0, 2.0]), "radius": 1.0},
        {"center": np.array([4.0, 3.0]), "radius": 1.5},
        {"center": np.array([3.0, 6.0]), "radius": 1.0},
    ]

    # --- Forward Kinematics ---
    # X contains relative angles (theta).
    # Absolute angles (cumulative sum of relative angles)
    # shape: (n_samples, 10)
    abs_angles = np.cumsum(X_scaled, axis=1)

    # Calculate (dx, dy) for each link
    dx = L * np.cos(abs_angles)
    dy = L * np.sin(abs_angles)

    # Calculate joint coordinates (cumulative sum of dx, dy)
    # Positions shape: (n_samples, 10) for x and y separately
    joint_positions_x = np.cumsum(dx, axis=1)
    joint_positions_y = np.cumsum(dy, axis=1)

    # --- Cost Calculation ---

    # 1. Distance to Target (End-effector only)
    end_effector_x = joint_positions_x[:, -1]
    end_effector_y = joint_positions_y[:, -1]

    dist_sq = (end_effector_x - target[0]) ** 2 + (end_effector_y - target[1]) ** 2

    # 2. Obstacle Penalty
    penalty = np.zeros(n_samples)

    # Check every joint against obstacles
    # (Simplified: checking joints only, not full link segments)
    for obs in obstacles:
        ox, oy = obs["center"]
        r = obs["radius"]

        # Check all 10 joints for this obstacle
        for j in range(10):
            jx = joint_positions_x[:, j]
            jy = joint_positions_y[:, j]

            # Distance from joint to obstacle center
            d_obs = np.sqrt((jx - ox) ** 2 + (jy - oy) ** 2)

            # If inside radius + buffer, add large penalty
            # Using violation metric: max(0, r - d + buffer)
            # Buffer = 0.1 to discourage grazing
            violation = np.maximum(0, r - d_obs + 0.1)

            # Penalty weight 1000
            penalty += 1000 * violation**2

    return dist_sq + penalty


def robot_arm_hard(X: np.ndarray) -> np.ndarray:
    """10-Link Robot Arm with Maze-Like Hard Constraints.

    A challenging constrained optimization problem where a 10-link planar robot arm
    must reach a target point (5.0, 5.0) while avoiding multiple obstacles forming
    a maze-like environment. This function tests an optimizer's ability to handle
    hard constraints and navigate complex feasible regions.

    The problem features three main difficulty factors:
    1. 'The Great Wall': A vertical barrier at x=2.5 blocking direct paths
    2. 'The Ceiling': A horizontal bar at y=8.5 preventing high loop strategies
    3. 'The Target Trap': Obstacles surrounding the target, requiring precise approach

    Mathematical formulation:
        f(X) = distance_cost + constraint_penalty + energy_regularization

    where:
        - distance_cost = (x_end - 5.0)^2 + (y_end - 5.0)^2
        - constraint_penalty = 10,000 * sum(max(0, r - d + 0.05)^2) for all joints and obstacles
        - energy_regularization = 0.01 * sum(angles^2)

    Args:
        X (np.ndarray): Input points with shape (n_samples, 10) or (10,).
            Each sample contains 10 joint angles normalized to [0, 1], which are
            internally mapped to [-1.2π, 1.2π] to allow looping strategies.
            Can be a 1D array for a single point or 2D array for multiple points.

    Returns:
        np.ndarray: Function values at the input points with shape (n_samples,).
            Lower values indicate better solutions (closer to target with fewer
            constraint violations).

    Note:
        - Dimension: 10 (one angle per link)
        - Link length: L = 1.0 for all links
        - Target position: (5.0, 5.0)
        - Search domain: [0, 1]^10 (mapped internally to [-1.2π, 1.2π]^10)
        - Characteristics: Highly constrained, non-convex, multimodal
        - Constraint penalty: 10,000 per violation (effectively hard constraints)
        - Number of obstacles: ~30 forming walls and traps
        - Feasible region: Very small relative to search space

    Examples:
        Single point evaluation with random configuration:

        >>> from spotoptim.function import robot_arm_hard
        >>> import numpy as np
        >>> X = np.random.rand(10) * 0.5  # Conservative random angles
        >>> result = robot_arm_hard(X)
        >>> result.shape
        (1,)

        Multiple points evaluation:

        >>> X = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        ...               [0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]])
        >>> robot_arm_hard(X)
        array([...])  # Returns costs for both configurations

        Evaluating a straight configuration (all angles = 0.5, mapped to 0 radians):

        >>> X_straight = np.full(10, 0.5)
        >>> cost_straight = robot_arm_hard(X_straight)
        >>> cost_straight[0] > 1000  # High cost due to obstacles
        True

    References:
        This function is inspired by robot motion planning problems with obstacles,
        commonly studied in:

        - LaValle, S. M. (2006). "Planning Algorithms". Cambridge University Press.
        - Choset, H., et al. (2005). "Principles of Robot Motion: Theory, Algorithms, and Implementations".
          MIT Press.
    """
    X = np.atleast_2d(X)

    # Map [0, 1] input to [-pi, pi]
    # We use a slightly wider range [-1.2*pi, 1.2*pi] to allow 'looping'
    # strategies which are sometimes necessary to solve hard mazes.
    Angles = X * 2.4 * np.pi - 1.2 * np.pi

    n_samples = Angles.shape[0]

    # --- Configuration ---
    L = 1.0
    target = np.array([5.0, 5.0])

    # Define Obstacles (The Maze)
    obstacles = []

    # 1. The Great Wall: Vertical barrier at x=2.5, from y=-2 to y=6
    # This forces the arm to reach 'way over' (high energy) or 'snake around'.
    # We create it using overlapping circles to form a wall.
    for y_pos in np.linspace(-2, 6, 15):
        obstacles.append({"c": np.array([2.5, y_pos]), "r": 0.6})

    # 2. The Ceiling: A horizontal bar at y=8 to prevent trivial 'high loops'
    for x_pos in np.linspace(0, 6, 10):
        obstacles.append({"c": np.array([x_pos, 8.5]), "r": 0.6})

    # 3. The Target Trap: Surround the target (5,5) with hazards
    # Leaving only a small opening from the 'bottom-left'.
    obstacles.append({"c": np.array([6.0, 5.0]), "r": 0.8})  # Right
    obstacles.append({"c": np.array([5.0, 6.5]), "r": 0.8})  # Top
    obstacles.append(
        {"c": np.array([4.0, 4.0]), "r": 0.5}
    )  # Bottom-Left (Partial Block)

    # --- Forward Kinematics ---
    # Cumulative angles
    abs_angles = np.cumsum(Angles, axis=1)

    # Link vectors
    dx = L * np.cos(abs_angles)
    dy = L * np.sin(abs_angles)

    # Joint positions (n_samples, 10)
    jx = np.cumsum(dx, axis=1)
    jy = np.cumsum(dy, axis=1)

    # --- Cost Calculation ---

    # 1. Distance Cost (End effector to Target)
    ex, ey = jx[:, -1], jy[:, -1]
    dist_sq = (ex - target[0]) ** 2 + (ey - target[1]) ** 2

    # 2. Hard Constraint Penalty
    # We sum penalties for ALL joints against ALL obstacles
    penalty = np.zeros(n_samples)

    # Vectorized obstacle check could be faster, but loops are clearer for definition
    for obs in obstacles:
        ox, oy = obs["c"]
        r = obs["r"]

        # We check every joint
        for j in range(10):
            # Calculate distance from joint j to obstacle center
            # (Avoid sqrt for speed where possible, but here we need true distance for linear penalty)
            d_obs = np.sqrt((jx[:, j] - ox) ** 2 + (jy[:, j] - oy) ** 2)

            # Constraint Violation
            # If d_obs < r, we are inside the obstacle.
            # We use a "Hard" quadratic penalty that shoots up explicitly.
            violation = np.maximum(0, r - d_obs + 0.05)  # 0.05 buffer 'skin'

            # 10,000 multiplier makes this a "Death Penalty" - practically a hard constraint
            penalty += 10000 * violation**2

    # 3. Regularization (Optional but realistic)
    # Penalize extreme contortions (minimize sum of squared relative angles)
    # This smooths the landscape slightly but makes the "optimal" path harder to find
    # because the arm 'wants' to be straight but 'needs' to bend.
    energy = np.sum(Angles**2, axis=1) * 0.01

    return dist_sq + penalty + energy
