import numpy as np
import pandas as pd
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from spotoptim.utils.stats import normalize_X
from spotoptim.sampling.lhs import rlh
from scipy.stats.qmc import LatinHypercube


def jd(X: np.ndarray, p: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes and counts the distinct p-norm distances between all pairs of points in X.
    It returns:
    1) A list of distinct distances (sorted), and
    2) A corresponding multiplicity array that indicates how often each distance occurs.

    Args:
        X (np.ndarray):
            A 2D array of shape (n, d) representing n points in d-dimensional space.
        p (float, optional):
            The distance norm to use. p=1 uses the Manhattan (L1) norm, while p=2 uses the
            Euclidean (L2) norm. Defaults to 1.0 (Manhattan norm).

    Returns:
        (np.ndarray, np.ndarray):
            A tuple (J, distinct_d), where:
            - distinct_d is a 1D float array of unique, sorted distances between points.
            - J is a 1D integer array that provides the multiplicity (occurrence count)
              of each distance in distinct_d.

    Notes:
        Many thanks to the original author of this code, A Sobester, for providing the original Matlab code
        under the GNU Licence. Original Matlab Code: Copyright 2007 A Sobester:
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
        >>> import numpy as np
        >>> from spotoptim.sampling.mm import jd
        >>> # A small 3-point set in 2D
        >>> X = np.array([[0.0, 0.0],
        ...               [1.0, 1.0],
        ...               [2.0, 2.0]])
        >>> J, distinct_d = jd(X, p=2.0)
        >>> print("Distinct distances:", distinct_d)
        >>> print("Occurrences:", J)
        # Possible output (using Euclidean norm):
        # Distinct distances: [1.41421356 2.82842712]
        # Occurrences: [1 1]
        # Explanation: Distances are sqrt(2) between consecutive points and 2*sqrt(2) for the farthest pair.
            Distinct distances: [1.41421356 2.82842712]
            Occurrences: [2 1]
    """
    n = X.shape[0]

    # Allocate enough space for all pairwise distances
    # (n*(n-1))/2 pairs for an n-point set
    pair_count = n * (n - 1) // 2
    d = np.zeros(pair_count, dtype=float)

    # Fill the distance array
    idx = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            # Compute the p-norm distance
            d[idx] = np.linalg.norm(X[i] - X[j], ord=p)
            idx += 1

    # Find unique distances and their multiplicities
    distinct_d = np.unique(d)
    J = np.zeros_like(distinct_d, dtype=int)
    for i, val in enumerate(distinct_d):
        J[i] = np.sum(d == val)

    return J, distinct_d


def mm(X1: np.ndarray, X2: np.ndarray, p: Optional[float] = 1.0) -> int:
    """
    Determines which of two sampling plans has better space-filling properties
    according to the Morris-Mitchell criterion.

    Args:
        X1 (np.ndarray): A 2D array representing the first sampling plan.
        X2 (np.ndarray): A 2D array representing the second sampling plan.
        p (float, optional): The distance metric. p=1 uses Manhattan (L1) distance,
            while p=2 uses Euclidean (L2). Defaults to 1.0.

    Returns:
        int:
            - 0 if both plans are identical or equally space-filling
            - 1 if X1 is more space-filling
            - 2 if X2 is more space-filling

    Notes:
        Many thanks to the original author of this code, A Sobester, for providing the original Matlab code
        under the GNU Licence. Original Matlab Code: Copyright 2007 A Sobester:
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
        >>> import numpy as np
        >>> from spotoptim.sampling.mm import mm
        >>> # Create two 3-point sampling plans in 2D
        >>> X1 = np.array([[0.0, 0.0],
        ...                [0.5, 0.5],
        ...                [0.0, 1.0]])
        >>> X2 = np.array([[0.1, 0.1],
        ...                [0.4, 0.6],
        ...                [0.1, 0.9]])
        >>> # Compare which plan has better space-filling (Morris-Mitchell)
        >>> better = mm(X1, X2, p=2.0)
        >>> print(better)
        # Prints either 0, 1, or 2 depending on which plan is more space-filling.
    """
    # Quick check if the sorted sets of points are identical
    # (mimicking MATLAB's sortrows check)
    X1_sorted = X1[np.lexsort(np.rot90(X1))]
    X2_sorted = X2[np.lexsort(np.rot90(X2))]
    if np.array_equal(X1_sorted, X2_sorted):
        return 0  # Identical sampling plans

    # Compute distance multiplicities for each plan
    J1, d1 = jd(X1, p)
    J2, d2 = jd(X2, p)
    m1, m2 = len(d1), len(d2)

    # Construct V1 and V2: alternate distance and negative multiplicity
    V1 = np.zeros(2 * m1)
    V1[0::2] = d1
    V1[1::2] = -J1

    V2 = np.zeros(2 * m2)
    V2[0::2] = d2
    V2[1::2] = -J2

    # Trim the longer vector to match the size of the shorter
    m = min(m1, m2)
    V1 = V1[:m]
    V2 = V2[:m]

    # Compare element-by-element:
    # c[i] = 1 if V1[i] > V2[i], 2 if V1[i] < V2[i], 0 otherwise.
    c = (V1 > V2).astype(int) + 2 * (V1 < V2).astype(int)

    if np.sum(c) == 0:
        # Equally space-filling
        return 0
    else:
        # The first non-zero entry indicates which plan is better
        idx = np.argmax(c != 0)
        return c[idx]


def mmphi(
    X: np.ndarray, q: Optional[float] = 2.0, p: Optional[float] = 1.0, verbosity=0
) -> float:
    """
    Calculates the Morris-Mitchell sampling plan quality criterion.

    Args:
        X (np.ndarray):
            A 2D array representing the sampling plan, where each row is a point in
            d-dimensional space (shape: (n, d)).
        q (float, optional):
            Exponent used in the computation of the metric. Defaults to 2.0.
        p (float, optional):
            The distance norm to use. For example, p=1 is Manhattan (L1),
            p=2 is Euclidean (L2). Defaults to 1.0.
        verbosity (int, optional):
            If set to 1, prints additional information about the computation.
            Defaults to 0 (no additional output).

    Returns:
        float:
            The space-fillingness metric Phiq. Larger values typically indicate a more
            space-filling plan according to the Morris-Mitchell criterion.

    Notes:
        Many thanks to the original author of this code, A Sobester, for providing the original Matlab code
        under the GNU Licence. Original Matlab Code: Copyright 2007 A Sobester:
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
        >>> import numpy as np
        >>> from spotoptim.sampling.mm import mmphi
        >>> # Simple 3-point sampling plan in 2D
        >>> X = np.array([
        ...     [0.0, 0.0],
        ...     [0.5, 0.5],
        ...     [1.0, 1.0]
        ... ])
        >>> # Calculate the space-fillingness metric with q=2, using Euclidean distances (p=2)
        >>> quality = mmphi(X, q=2, p=2)
        >>> print(quality)
        # This value indicates how well points are spread out, with smaller being better.
    """
    # check that X has unique rows
    if X.shape[0] != len(np.unique(X, axis=0)):
        # issue a warning if there are duplicate rows
        print(
            "Warning: X contains duplicate rows. This may affect the space-fillingness metric."
        )
        # make X unique
        X = np.unique(X, axis=0)
    # Compute the distance multiplicities: J, and unique distances: d
    J, d = jd(X, p)
    print(f"J: {J}, d: {d}") if verbosity > 0 else None

    # Summation of J[i] * d[i]^(-q), then raised to 1/q
    # This follows the Morris-Mitchell definition.
    Phiq = np.sum(J * (d ** (-q))) ** (1.0 / q)
    return Phiq


def mmsort(X3D: np.ndarray, p: Optional[float] = 1.0) -> np.ndarray:
    """
    Ranks multiple sampling plans stored in a 3D array according to the
    Morris-Mitchell criterion, using a simple bubble sort.

    Args:
        X3D (np.ndarray):
            A 3D NumPy array of shape (n, d, m), where m is the number of
            sampling plans, and each plan is an (n, d) matrix of points.
        p (float, optional):
            The distance metric to use. p=1 for Manhattan (L1), p=2 for
            Euclidean (L2). Defaults to 1.0.

    Returns:
        np.ndarray:
            A 1D integer array of length m that holds the plan indices in
            ascending order of space-filling quality. The first index in the
            returned array corresponds to the most space-filling plan.

    Notes:
        Many thanks to the original author of this code, A Sobester, for providing the original Matlab code
        under the GNU Licence. Original Matlab Code: Copyright 2007 A Sobester:
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
        >>> import numpy as np
        >>> from spotoptim.sampling.mm import mmsort
        >>> # Suppose we have two 3-point sampling plans in 2D, stored in X3D:
        >>> X1 = np.array([[0.0, 0.0],
        ...                [0.5, 0.5],
        ...                [1.0, 1.0]])
        >>> X2 = np.array([[0.2, 0.2],
        ...                [0.6, 0.4],
        ...                [0.9, 0.9]])
        >>> # Stack them along the third dimension: shape will be (3, 2, 2)
        >>> X3D = np.stack([X1, X2], axis=2)
        >>> # Sort them using the Morris-Mitchell criterion with p=2
        >>> ranking = mmsort(X3D, p=2.0)
        >>> print(ranking)
        # It might print [2 1] or [1 2], depending on which plan is more space-filling.
    """
    # Number of plans (m)
    m = X3D.shape[2]

    # Create index array (1-based to match original MATLAB convention)
    Index = np.arange(1, m + 1)

    swap_flag = True
    while swap_flag:
        swap_flag = False
        i = 0
        while i < m - 1:
            # Compare plan at Index[i] vs. Index[i+1] using mm()
            # Note: subtract 1 from each index to convert to 0-based array indexing
            if mm(X3D[:, :, Index[i] - 1], X3D[:, :, Index[i + 1] - 1], p) == 2:
                # Swap indices if the second plan is more space-filling
                Index[i], Index[i + 1] = Index[i + 1], Index[i]
                swap_flag = True
            i += 1

    return Index


def perturb(X: np.ndarray, PertNum: Optional[int] = 1) -> np.ndarray:
    """
    Performs a specified number of random element swaps on a sampling plan.
    If the plan is a Latin hypercube, the result remains a valid Latin hypercube.

    Args:
        X (np.ndarray):
            A 2D array (sampling plan) of shape (n, k), where each row is a point
            and each column is a dimension.
        PertNum (int, optional):
            The number of element swaps (perturbations) to perform. Defaults to 1.

    Returns:
        np.ndarray:
            The perturbed sampling plan, identical in shape to the input, with
            one or more random column swaps executed.

    Notes:
        Many thanks to the original author of this code, A Sobester, for providing the original Matlab code
        under the GNU Licence. Original Matlab Code: Copyright 2007 A Sobester:
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
        >>> import numpy as np
        >>> from spotoptim.sampling.mm import perturb
        >>> # Create a simple 4x2 sampling plan
        >>> X_original = np.array([
        ...     [1, 3],
        ...     [2, 4],
        ...     [3, 1],
        ...     [4, 2]
        ... ])
        >>> # Perturb it once
        >>> X_perturbed = perturb(X_original, PertNum=1)
        >>> print(X_perturbed)
        # The output may differ due to random swaps, but each column is still a permutation of [1,2,3,4].
            [[1 3]
            [2 2]
            [3 1]
            [4 4]]
    """
    # Get dimensions of the plan
    n, k = X.shape
    if n < 2 or k < 2:
        raise ValueError("Latin hypercubes require at least 2 points and 2 dimensions")

    for _ in range(PertNum):
        # Pick a random column
        col = int(np.floor(np.random.rand() * k))

        # Pick two distinct row indices
        el1, el2 = 0, 0
        while el1 == el2:
            el1 = int(np.floor(np.random.rand() * n))
            el2 = int(np.floor(np.random.rand() * n))

        # Swap the two selected elements in the chosen column
        X[el1, col], X[el2, col] = X[el2, col], X[el1, col]

    return X


def mmlhs(
    X_start: np.ndarray,
    population: int,
    iterations: int,
    q: Optional[float] = 2.0,
    plot=False,
) -> np.ndarray:
    """
    Performs an evolutionary search (using perturbations) to find a Morris-Mitchell
    optimal Latin hypercube, starting from an initial plan X_start.

    This function does the following:
      1. Initializes a "best" Latin hypercube (X_best) from the provided X_start.
      2. Iteratively perturbs X_best to create offspring.
      3. Evaluates the space-fillingness of each offspring via the Morris-Mitchell
         metric (using mmphi).
      4. Updates the best plan whenever a better offspring is found.

    Args:
        X_start (np.ndarray):
            A 2D array of shape (n, k) providing the initial Latin hypercube
            (n points in k dimensions).
        population (int):
            Number of offspring to create in each generation.
        iterations (int):
            Total number of generations to run the evolutionary search.
        q (float, optional):
            The exponent used by the Morris-Mitchell space-filling criterion.
            Defaults to 2.0.
        plot (bool, optional):
            If True, a simple scatter plot of the first two dimensions will be
            displayed at each iteration. Only if k >= 2. Defaults to False.

    Returns:
        np.ndarray:
            A 2D array representing the most space-filling Latin hypercube found
            after all iterations, of the same shape as X_start.

    Notes:
        Many thanks to the original author of this code, A Sobester, for providing the original Matlab code
        under the GNU Licence. Original Matlab Code: Copyright 2007 A Sobester:
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
        >>> import numpy as np
        >>> from spotoptim.sampling.mm import mmlhs
        >>> # Suppose we have an initial 4x2 plan
        >>> X_start = np.array([
        ...     [0, 0],
        ...     [1, 3],
        ...     [2, 1],
        ...     [3, 2]
        ... ])
        >>> # Search for a more space-filling plan
        >>> X_opt = mmlhs(X_start, population=5, iterations=10, q=2)
        >>> print("Optimized plan:")
        >>> print(X_opt)
    """
    n = X_start.shape[0]
    if n < 2:
        raise ValueError("Latin hypercubes require at least 2 points")
    k = X_start.shape[1]
    if k < 2:
        raise ValueError("Latin hypercubes are not defined for dim k < 2")

    # Initialize best plan and its metric
    X_best = X_start.copy()
    Phi_best = mmphi(X_best, q=q)

    # After 85% of iterations, reduce the mutation rate to 1
    leveloff = int(np.floor(0.85 * iterations))

    for it in range(1, iterations + 1):
        # Decrease number of mutations over time
        if it < leveloff:
            mutations = int(round(1 + (0.5 * n - 1) * (leveloff - it) / (leveloff - 1)))
        else:
            mutations = 1

        X_improved = X_best.copy()
        Phi_improved = Phi_best

        # Create offspring, evaluate, and keep the best
        for _ in range(population):
            X_try = perturb(X_best.copy(), mutations)
            Phi_try = mmphi(X_try, q=q)

            if Phi_try < Phi_improved:
                X_improved = X_try
                Phi_improved = Phi_try

        # Update the global best if we found a better plan
        if Phi_improved < Phi_best:
            X_best = X_improved
            Phi_best = Phi_improved

        # Simple visualization of the first two dimensions
        if plot and (X_best.shape[1] >= 2):
            plt.clf()
            plt.scatter(X_best[:, 0], X_best[:, 1], marker="o")
            plt.grid(True)
            plt.title(f"Iteration {it} - Current Best Plan")
            plt.pause(0.01)

    return X_best


def phisort(
    X3D: np.ndarray, q: Optional[float] = 2.0, p: Optional[float] = 1.0
) -> np.ndarray:
    """
    Ranks multiple sampling plans stored in a 3D array by the Morris-Mitchell
    numerical quality metric (mmphi). Uses a simple bubble-sort:
    sampling plans with smaller mmphi values are placed first in the index array.

    Args:
        X3D (np.ndarray):
            A 3D array of shape (n, d, m), where m is the number of sampling plans.
        q (float, optional):
            Exponent for the mmphi metric. Defaults to 2.0.
        p (float, optional):
            Distance norm for mmphi. p=1 is Manhattan; p=2 is Euclidean. Defaults to 1.0.

    Returns:
        np.ndarray:
            A 1D integer array of length m, giving the plan indices in ascending
            order of mmphi. The first index in the returned array corresponds
            to the numerically lowest mmphi value.

    Notes:
        Many thanks to the original author of this code, A Sobester, for providing the original Matlab code
        under the GNU Licence. Original Matlab Code: Copyright 2007 A Sobester:
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
        >>> import numpy as np
            from spotoptim.sampling.mm import phisort
            from spotoptim.sampling.mm import bestlh
            X1 = bestlh(n=5, k=2, population=5, iterations=10)
            X2 = bestlh(n=5, k=2, population=15, iterations=20)
            X3 = bestlh(n=5, k=2, population=25, iterations=30)
            # Map X1 and X2 so that X3D has the two sampling plans in X3D[:, :, 0] and X3D[:, :, 1]
            X3D = np.array([X1, X2])
            print(phisort(X3D))
            X3D = np.array([X3, X2])
            print(phisort(X3D))
                [2 1]
                [1 2]
    """
    # Number of 2D sampling plans
    m = X3D.shape[2]

    # Create a 1-based index array
    Index = np.arange(1, m + 1)

    # Bubble-sort: plan with lower mmphi() climbs toward the front
    swap_flag = True
    while swap_flag:
        swap_flag = False
        for i in range(m - 1):
            # Retrieve mmphi values for consecutive plans
            val_i = mmphi(X3D[:, :, Index[i] - 1], q=q, p=p)
            val_j = mmphi(X3D[:, :, Index[i + 1] - 1], q=q, p=p)

            # Swap if the left plan's mmphi is larger (i.e. 'worse')
            if val_i > val_j:
                Index[i], Index[i + 1] = Index[i + 1], Index[i]
                swap_flag = True

    return Index


def subset(X: np.ndarray, ns: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns a space-filling subset of a given size from a sampling plan, along with
    the remainder. It repeatedly attempts to substitute each point in the subset
    with a point from the remainder if doing so improves the Morris-Mitchell metric.

    Args:
        X (np.ndarray):
            A 2D array representing the original sampling plan, of shape (n, d).
        ns (int):
            The size of the desired subset.

    Returns:
        (np.ndarray, np.ndarray):
            A tuple (Xs, Xr) where:
            - Xs is the chosen subset of size ns, with space-filling properties.
            - Xr is the remainder (X \\ Xs).

    Notes:
        Many thanks to the original author of this code, A Sobester, for providing the original Matlab code
        under the GNU Licence. Original Matlab Code: Copyright 2007 A Sobester:
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
        >>> from spotoptim.sampling.mm import subset, bestlh
            X = bestlh(n=5, k=3, population=5, iterations=10)
            Xs, Xr = subset(X, ns=2)
            print(Xs)
            print(Xr)
                [[0.25 0.   0.5 ]
                [0.5  0.75 0.  ]]
                [[1.   0.25 0.25]
                [0.   1.   0.75]
                [0.75 0.5  1.  ]]
    """
    # Number of total points
    n = X.shape[0]

    # Morris-Mitchell parameters
    p = 1
    q = 5

    # Create a random permutation of row indices
    r = np.random.permutation(n)

    # Initial subset and remainder
    Xs = X[r[:ns], :].copy()
    Xr = X[r[ns:], :].copy()

    # Attempt to improve space-filling by swapping points
    for j in range(ns):
        orig_crit = mmphi(Xs, q=q, p=p)
        orig_point = Xs[j, :].copy()

        # Track best substitution index and metric
        bestsub = 0
        bestsubcrit = np.inf

        # Try replacing Xs[j] with each candidate in Xr
        for i in range(n - ns):
            Xs[j, :] = Xr[i, :]
            crit = mmphi(Xs, q=q, p=p)
            if crit < bestsubcrit:
                bestsubcrit = crit
                bestsub = i

        # If a better subset is found, swap permanently
        if bestsubcrit < orig_crit:
            Xs[j, :] = Xr[bestsub, :].copy()
            Xr[bestsub, :] = orig_point
        else:
            Xs[j, :] = orig_point

    return Xs, Xr


def mmphi_intensive(
    X: np.ndarray,
    q: Optional[float] = 2.0,
    p: Optional[float] = 2.0,
    normalize_flag: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Calculates a size-invariant Morris-Mitchell criterion.

    This "intensive" version of the criterion allows for the comparison of
    sampling plans with different sample sizes by normalizing for the number
    of point pairs. A smaller value indicates a better (more space-filling)
    design.

    Args:
        X (np.ndarray):
            A 2D array representing the sampling plan (shape: (n, d)).
        q (float, optional):
            The exponent used in the computation of the metric. Defaults to 2.0.
        p (float, optional):
            The distance norm to use (e.g., p=1 for Manhattan, p=2 for Euclidean).
            Defaults to 2.0.
        normalize_flag (bool, optional):
            If True, normalizes the X array before computing distances.
            Defaults to False.

    Returns:
        tuple[float, np.ndarray, np.ndarray]:
            A tuple containing:
            - intensive_phiq: The intensive space-fillingness metric.
            - J: Multiplicities of distances.
            - d: Unique distances.

    Examples:
        >>> import numpy as np
        >>> from spotoptim.sampling.mm import mmphi_intensive
        >>> # Create a simple 3-point sampling plan in 2D
        >>> X = np.array([
        ...     [0.0, 0.0],
        ...     [0.5, 0.5],
        ...     [1.0, 1.0]
        ... ])
        >>> # Calculate the intensive space-fillingness metric with q=2, using Euclidean distances (p=2)
        >>> quality, J, d = mmphi_intensive(X, q=2, p=2)
        >>> print(quality)
    """
    # Ensure there are no duplicate points
    if X.shape[0] != len(np.unique(X, axis=0)):
        X = np.unique(X, axis=0)

    n_points = X.shape[0]

    # Normalize X to [0, 1] in each dimension if requested
    if normalize_flag:
        X = normalize_X(X)

    # The criterion is not well-defined for fewer than 2 points.
    if n_points < 2:
        return np.inf, 0, 0

    # Get the unique distances and their multiplicities
    J, d = jd(X, p=p)

    # If all points are identical, the design is infinitely bad.
    if d.size == 0:
        return np.inf, J, d

    # Calculate the number of unique pairs of points
    M = n_points * (n_points - 1) / 2

    try:
        # Calculate the sum term of the original mmphi
        sum_term = np.sum(J * (d ** (-q)))
        # Normalize the sum by M before taking the final root
        intensive_phiq = (sum_term / M) ** (1.0 / q)
    except ZeroDivisionError:
        return np.inf
    except FloatingPointError:
        return np.inf
    except Exception:
        return np.inf

    return intensive_phiq, J, d


def mmphi_intensive_update(
    X: np.ndarray,
    new_point: np.ndarray,
    J: np.ndarray,
    d: np.ndarray,
    q: float = 2.0,
    p: float = 2.0,
    normalize_flag: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Updates the Morris-Mitchell intensive criterion for n+1 points by adding a new point to the design.
    This should be more efficient than recalculating the metric from scratch, because it only needs to
    compute the distances between the new point and the existing points.

    Args:
        X (np.ndarray): Existing sampling plan (shape: (n, d)).
        new_point (np.ndarray): New point to add (shape: (d,)).
        J (np.ndarray): Multiplicities of distances for the existing design.
        d (np.ndarray): Unique distances for the existing design.
        q (float): Exponent used in the computation of the Morris-Mitchell metric. Defaults to 2.0.
        p (float): Distance norm to use (e.g., p=1 for Manhattan, p=2 for Euclidean). Defaults to 2.0.
        normalize_flag (bool): If True, normalizes the X array and the new_point before computing distances. Defaults to False.

    Returns:
        tuple[float, np.ndarray, np.ndarray]: Updated intensive_phiq, updated_J, updated_d.

    Examples:
        >>> import numpy as np
        >>> from spotoptim.sampling.mm import mmphi_intensive_update
        >>> # Existing design with 3 points in 2D
        >>> X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        >>> phiq, J, d = mmphi_intensive(X, q=2, p=2)
        >>> # New point to add
        >>> new_point = np.array([0.1, 0.1])
        >>> # Update the intensive criterion
        >>> updated_phiq, updated_J, updated_d = mmphi_intensive_update(X, new_point, J, d, q=2, p=2)

    """
    n_points = X.shape[0]
    if n_points < 1:
        raise ValueError("The existing design must contain at least one point.")

    # Normalize X and new_point to [0, 1] in each dimension if requested
    if normalize_flag:
        X = normalize_X(X)
        new_point = (new_point - np.min(X, axis=0)) / (
            np.max(X, axis=0) - np.min(X, axis=0)
        )

    # Compute distances between the new point and all existing points
    new_distances = np.array(
        [np.linalg.norm(new_point - X[i], ord=p) for i in range(n_points)]
    )

    # Combine old distances and new distances into a single list
    all_distances = []
    for dist, count in zip(d, J):
        all_distances.extend([dist] * count)
    all_distances.extend(new_distances)

    # Find unique distances and their counts
    updated_d, updated_J = np.unique(all_distances, return_counts=True)

    # Calculate the number of unique pairs of points
    M = (n_points + 1) * n_points / 2

    # Compute the updated intensive_phiq
    sum_term = np.sum(updated_J * (updated_d ** (-q)))
    intensive_phiq = (sum_term / M) ** (1.0 / q)

    return intensive_phiq, updated_J, updated_d


def propose_mmphi_intensive_minimizing_point(
    X: np.ndarray,
    n_candidates: int = 1000,
    q: float = 2.0,
    p: float = 2.0,
    seed: Optional[int] = None,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    normalize_flag: bool = False,
) -> np.ndarray:
    """
    Propose a new point that, when added to X, minimizes the intensive Morris-Mitchell (mmphi_intensive) criterion.

    Args:
        X (np.ndarray): Existing points, shape (n_points, n_dim).
        n_candidates (int): Number of random candidates to sample.
        q (float): Exponent for mmphi_intensive.
        p (float): Distance norm for mmphi_intensive.
        seed (int, optional): Random seed.
        lower (np.ndarray, optional): Lower bounds for each dimension (default: 0).
        upper (np.ndarray, optional): Upper bounds for each dimension (default: 1).
        normalize_flag (bool): If True, normalizes the X array and candidate points before computing distances. Defaults to False.

    Returns:
        np.ndarray: Proposed new point, shape (1, n_dim).

    Examples:
        >>> import numpy as np
            from spotoptim.sampling.mm import propose_mmphi_intensive_minimizing_point
            # Existing design with 3 points in 2D
            X = np.array([[1.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
            # Propose a new point
            new_point = propose_mmphi_intensive_minimizing_point(X, n_candidates=500, q=2, p=2, seed=42)
            print(new_point)
            # plot the existing points and the new proposed point
            import matplotlib.pyplot as plt
            plt.scatter(X[:, 0], X[:, 1], color='blue', label='Existing Points')
            plt.scatter(new_point[0, 0], new_point[0, 1], color='red', label='Proposed Point')
            plt.legend()
            # add grid and labels
            plt.grid()
            plt.title('MM-PHI Proposed Point')
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.show()
    """
    rng = np.random.default_rng(seed)
    n_dim = X.shape[1]
    if lower is None:
        lower = np.zeros(n_dim)
    if upper is None:
        upper = np.ones(n_dim)
    if np.any(lower >= upper):
        raise ValueError("Lower bounds must be less than upper bounds.")
    # Generate candidate points uniformly
    candidates = rng.uniform(lower, upper, size=(n_candidates, n_dim))
    if normalize_flag:
        X = normalize_X(X)
        candidates = (candidates - lower) / (upper - lower)
    best_phi = np.inf
    best_point = None
    for cand in candidates:
        X_aug = np.vstack([X, cand])
        phi, _, _ = mmphi_intensive(X_aug, q=q, p=p)
        if phi < best_phi:
            best_phi = phi
            best_point = cand
    return best_point.reshape(1, -1)


def mm_improvement(
    x,
    X_base,
    phi_base=None,
    J_base=None,
    d_base=None,
    q=2,
    p=2,
    normalize_flag=False,
    verbose=False,
    exponential=True,
) -> float:
    """
    Calculates the Morris-Mitchell improvement for a candidate point x.

    Args:
        x (np.ndarray): Candidate point (1D array).
        X_base (np.ndarray): Existing design points.
        J_base (np.ndarray): Multiplicities of distances for X_base.
        d_base (np.ndarray): Unique distances for X_base.
        q (int): Number of nearest neighbors for MM metric.
        p (int): Power for MM metric.
        normalize_flag (bool): If True, normalizes the X array and candidate point before computing distances. Defaults to False.
        exponential (bool): If True, the exponential is applied.

    Returns:
        float: Morris-Mitchell improvement.

    Examples:
        >>> import numpy as np
        >>> from spotoptim.sampling.mm import mm_improvement
        >>> X_base = np.array([[0.1, 0.2], [0.4, 0.5], [0.7, 0.8]])
        >>> x = np.array([0.5, 0.5])
        >>> improvement = mm_improvement(x, X_base, q=2, p=2)
        >>> print(improvement)
        0.123456789
    """
    if phi_base is None or J_base is None or d_base is None:
        phi_base, J_base, d_base = mmphi_intensive(
            X_base, q=q, p=p, normalize_flag=normalize_flag
        )
    phi_new, _, _ = mmphi_intensive_update(
        X_base, x, J_base, d_base, q=q, p=p, normalize_flag=normalize_flag
    )
    if exponential:
        y_mm = np.exp(phi_base - phi_new)
    else:
        y_mm = phi_base - phi_new
    if verbose:
        print(f"Morris-Mitchell base: {phi_base}")
        print(f"Morris-Mitchell new: {phi_new}")
        print(f"Morris-Mitchell improvement: {y_mm}")
    return float(y_mm)


def bestlh(
    n: int,
    k: int,
    population: int,
    iterations: int,
    p=1,
    plot=False,
    verbosity=0,
    edges=0,
    q_list=[1, 2, 5, 10, 20, 50, 100],
) -> np.ndarray:
    """
    Generates an optimized Latin hypercube by evolving the Morris-Mitchell
    criterion across multiple exponents (q values) and selecting the best plan.

    Args:
        n (int):
            Number of points required in the Latin hypercube.
        k (int):
            Number of design variables (dimensions).
        population (int):
            Number of offspring in each generation of the evolutionary search.
        iterations (int):
            Number of generations for the evolutionary search.
        p (int, optional):
            The distance norm to use. p=1 for Manhattan (L1), p=2 for Euclidean (L2).
            Defaults to 1 (faster than 2).
        plot (bool, optional):
            If True, a scatter plot of the optimized plan in the first two dimensions
            will be displayed. Only if k>=2. Defaults to False.
        verbosity (int, optional):
            Verbosity level. 0 is silent, 1 prints the best q value found. Defaults to 0.
        edges (int, optional):
            If 1, places centers of the extreme bins at the domain edges ([0,1]).
            Otherwise, bins are fully contained within the domain, i.e. midpoints.
            Defaults to 0.
        q_list (list, optional):
            A list of q values to optimize. Defaults to [1, 2, 5, 10, 20, 50, 100].
            These values are used to evaluate the space-fillingness of the Latin
            hypercube. The best plan is selected based on the lowest mmphi value.

    Returns:
        np.ndarray:
            A 2D array of shape (n, k) representing an optimized Latin hypercube.

    Examples:
        >>> import numpy as np
        >>> from spotoptim.sampling.mm import bestlh
        # Generate a 5-point, 2-dimensional Latin hypercube
        >>> X = bestlh(n=5, k=2, population=5, iterations=10)
        >>> print(X.shape)
        (5, 2)
    """
    if n < 2:
        raise ValueError("Latin hypercubes require at least 2 points")
    if k < 2:
        raise ValueError("Latin hypercubes are not defined for dim k < 2")

    # Start with a random Latin hypercube
    X_start = rlh(n, k, edges=edges)

    # Allocate a 3D array to store the results for each q
    # (shape: (n, k, number_of_q_values))
    X3D = np.zeros((n, k, len(q_list)))

    # Evolve the plan for each q in q_list
    for i, q_val in enumerate(q_list):
        if verbosity > 0:
            print(f"Now optimizing for q={q_val}...")
        X3D[:, :, i] = mmlhs(X_start, population, iterations, q_val)

    # Sort the set of evolved plans according to the Morris-Mitchell criterion
    index_order = mmsort(X3D, p=p)

    # index_order is a 1-based array of plan indices; the first element is the best
    best_idx = index_order[0] - 1
    if verbosity > 0:
        print(f"Best lh found using q={q_list[best_idx]}...")

    # The best plan in 3D array order
    X = X3D[:, :, best_idx]

    # Plot the first two dimensions
    if plot and (k >= 2):
        plt.scatter(X[:, 0], X[:, 1], c="r", marker="o")
        plt.title(f"Morris-Mitchell optimum plan found using q={q_list[best_idx]}")
        plt.xlabel("x_1")
        plt.ylabel("x_2")
        plt.grid(True)
        plt.show()

    return X


def plot_mmphi_vs_n_lhs(
    k_dim: int,
    seed: int,
    n_min: int = 10,
    n_max: int = 100,
    n_step: int = 5,
    q_phi: float = 2.0,
    p_phi: float = 2.0,
) -> None:
    """
    Generates LHS designs for varying n, calculates mmphi and mmphi_intensive,
    and plots them against the number of samples (n).

    Args:
        k_dim (int): Number of dimensions for the LHS design.
        seed (int): Random seed for reproducibility.
        n_min (int): Minimum number of samples.
        n_max (int): Maximum number of samples.
        n_step (int): Step size for increasing n.
        q_phi (float): Exponent q for the Morris-Mitchell criteria.
        p_phi (float): Distance norm p for the Morris-Mitchell criteria.

    Returns:
        None: Displays a plot of mmphi and mmphi_intensive vs. number of samples (n).

    Examples:
        >>> from spotoptim.sampling.mm import plot_mmphi_vs_n_lhs
        >>> plot_mmphi_vs_n_lhs(k_dim=3, seed=42, n_min=10, n_max=50, n_step=5, q_phi=2.0, p_phi=2.0)
    """
    n_values = list(range(n_min, n_max + 1, n_step))
    if not n_values:
        print("Warning: n_values list is empty. Check n_min, n_max, and n_step.")
        return
    mmphi_results = []
    mmphi_intensive_results = []
    lhs_sampler = LatinHypercube(d=k_dim, seed=seed)

    for n_points in n_values:
        if n_points < 2:  # mmphi requires at least 2 points to calculate distances
            print(f"Skipping n={n_points} as it's less than 2.")
            mmphi_results.append(np.nan)
            mmphi_intensive_results.append(np.nan)
            continue
        try:
            X_design = lhs_sampler.random(n=n_points)
            phi = mmphi(X_design, q=q_phi, p=p_phi)
            phi_intensive, _, _ = mmphi_intensive(X_design, q=q_phi, p=p_phi)
            mmphi_results.append(phi)
            mmphi_intensive_results.append(phi_intensive)
        except Exception as e:
            print(f"Error calculating for n={n_points}: {e}")
            mmphi_results.append(np.nan)
            mmphi_intensive_results.append(np.nan)

    fig, ax1 = plt.subplots(figsize=(9, 6))

    color = "tab:red"
    ax1.set_xlabel("Number of Samples (n)")
    ax1.set_ylabel("mmphi (Phiq)", color=color)
    ax1.plot(
        n_values,
        mmphi_results,
        color=color,
        marker="o",
        linestyle="-",
        label="mmphi (Phiq)",
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.7)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = "tab:blue"
    ax2.set_ylabel(
        "mmphi_intensive (PhiqI)", color=color
    )  # we already handled the x-label with ax1
    ax2.plot(
        n_values,
        mmphi_intensive_results,
        color=color,
        marker="x",
        linestyle="--",
        label="mmphi_intensive (PhiqI)",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title(
        f"Morris-Mitchell Criteria vs. Number of Samples (n)\nLHS (k={k_dim}, q={q_phi}, p={p_phi})"
    )
    # Add legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="best")
    plt.show()


def plot_mmphi_vs_points(
    X_base: np.ndarray,
    x_min: np.ndarray,
    x_max: np.ndarray,
    p_min: int = 10,
    p_max: int = 100,
    p_step: int = 10,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """
    Plot the Morris-Mitchell criterion versus the number of added points.

    Args:
        X_base (np.ndarray): Base design matrix
        x_min (np.ndarray): Lower bounds for variables
        x_max (np.ndarray): Upper bounds for variables
        p_min (int): Minimum number of points to add
        p_max (int): Maximum number of points to add
        p_step (int): Step size for number of points
        n_repeats (int): Number of repetitions for each point count

    Returns:
        pd.DataFrame: Summary DataFrame with mean and std of mmphi for each number of added points.

    Examples:
        >>> import numpy as np
        >>> from spotoptim.sampling.mm import plot_mmphi_vs_points
        >>> # Define base design
        >>> X_base = np.array([[0.1, 0.2], [0.4, 0.5], [0.7, 0.8]])
        >>> # Define variable bounds
        >>> x_min = np.array([0.0, 0.0])
        >>> x_max = np.array([1.0, 1.0])
        >>> # Plot mmphi vs number of added points
        >>> df_summary = plot_mmphi_vs_points(X_base, x_min, x_max, p_min=10, p_max=50, p_step=10, n_repeats=3)
    """
    n, m = X_base.shape
    p_values = range(p_min, p_max + 1, p_step)

    # Calculate base mmphi value
    mmphi_base, _, _ = mmphi_intensive(X=X_base)

    # Store results
    results = []

    # For each number of points
    for p in p_values:
        # Repeat multiple times to get average behavior
        for _ in range(n_repeats):
            # Generate random points
            x_random = np.random.uniform(low=x_min, high=x_max, size=(p, m))

            # Append to base design
            X_extended = np.append(X_base, x_random, axis=0)

            # Calculate new mmphi value
            mmphi_extended, _, _ = mmphi_intensive(X=X_extended)

            # Store results
            results.append({"n_points": p, "mmphi": mmphi_extended})

    # Convert results to DataFrame for easier plotting
    df_results = pd.DataFrame(results)

    # Calculate mean and std for each point count
    df_summary = (
        df_results.groupby("n_points").agg({"mmphi": ["mean", "std"]}).reset_index()
    )

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot mean line with error bars
    plt.errorbar(
        df_summary["n_points"],
        df_summary["mmphi"]["mean"],
        yerr=df_summary["mmphi"]["std"],
        fmt="bo-",
        capsize=5,
        capthick=1,
        elinewidth=1,
        label="Mean mmphi ± std",
    )

    # Add error bands (optional - can keep or remove depending on preference)
    plt.fill_between(
        df_summary["n_points"],
        df_summary["mmphi"]["mean"] - df_summary["mmphi"]["std"],
        df_summary["mmphi"]["mean"] + df_summary["mmphi"]["std"],
        alpha=0.1,
        color="blue",
    )

    # Add baseline
    plt.axhline(
        y=mmphi_base,
        color="r",
        linestyle="--",
        label=f"Base design mmphi ({mmphi_base:.4f})",
    )

    # Customize plot
    plt.xlabel("Number of Added Points")
    plt.ylabel("Morris-Mitchell Criterion")
    plt.title("Morris-Mitchell Criterion vs Number of Added Points")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Show plot
    plt.show()
    plt.close()

    return df_summary


def mm_improvement_contour(
    X_base, x1=np.linspace(0, 1, 100), x2=np.linspace(0, 1, 100), q=2, p=2
):
    """
    Generates a contour plot of the Morris-Mitchell improvement over a grid defined by x1 and x2.

    Args:
        X_base (np.ndarray):
            Base design points.
        x1 (np.ndarray):
            Grid values for the first dimension. Default is np.linspace(0, 1, 100).
        x2 (np.ndarray): Grid values for the second dimension. Default is np.linspace(0, 1, 100).
        q (int):
            Morris-Mitchell metric parameter. Default is 2.
        p (int):
            Morris-Mitchell metric parameter. Default is 2.
    Returns:
        None: Displays a contour plot of the Morris-Mitchell improvement.

    Examples:
        >>> import numpy as np
            from spotoptim.sampling.mm import mm_improvement_contour
            X_base = np.array([[0.1, 0.1], [0.2, 0.2], [0.7, 0.7]])
            mm_improvement_contour(X_base)
    """

    _, J_base, d_base = mmphi_intensive(X_base, q=2, p=2)
    X1, X2 = np.meshgrid(x1, x2)
    improvement_grid = np.zeros(X1.shape)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x = np.array([X1[i, j], X2[i, j]])
            improvement_grid[i, j] = mm_improvement(x, X_base, J_base, d_base, q=2, p=2)
    plt.contourf(X1, X2, improvement_grid, levels=30, cmap="viridis")
    plt.colorbar(label="Morris-Mitchell Improvement")
    plt.scatter(X_base[:, 0], X_base[:, 1], color="red", label="X_base")
    plt.title("Morris-Mitchell Improvement Contour Plot")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()
