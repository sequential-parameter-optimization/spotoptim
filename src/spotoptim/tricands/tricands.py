import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull
import numpy as np


def tricands_interior(X: np.ndarray) -> dict:
    """
    Generate interior candidates using Delaunay triangulation.

    Subroutine used by tricands wrapper.

    Args:
        X (np.ndarray): Input design matrix of shape (n_samples, n_features).

    Returns:
        dict: A dictionary containing:
            - 'cand' (np.ndarray): Candidate points (midpoints of triangles).
            - 'tri' (np.ndarray): Simplicies of the Delaunay triangulation.

    Raises:
        Exception: If the number of points is less than n_features + 1.
    """
    m = X.shape[1]
    n = X.shape[0]
    if n < m + 1:
        raise Exception("must have nrow(X) >= ncol(X) + 1")

    # possible to further vectorize?
    # find the middle of triangles
    tri = Delaunay(X, qhull_options="Q12").simplices
    Xcand = np.zeros([tri.shape[0], m])
    for i in range(tri.shape[0]):
        Xcand[i, :] = np.mean(X[tri[i,],], axis=0)

    return {"cand": Xcand, "tri": tri}


def tricands_fringe(
    X: np.ndarray, p: float = 0.5, lower: float = 0, upper: float = 1
) -> dict:
    """
    Generate fringe candidates outside the convex hull.

    Subroutine used by tricands wrapper. Assumes a bounding box of [lower, upper]^m.

    Args:
        X (np.ndarray): Input design matrix of shape (n_samples, n_features).
        p (float, optional): Distance to the boundary (0 = on hull, 1 = on boundary). Defaults to 0.5.
        lower (float, optional): Lower bound of bounding box for all dimensions. Defaults to 0.
        upper (float, optional): Upper bound of bounding box for all dimensions. Defaults to 1.

    Returns:
        dict: A dictionary containing:
            - 'XF' (np.ndarray): Fringe candidate points.
            - 'XB' (np.ndarray): Boundary points (means of external facets).
            - 'qhull' (scipy.spatial.ConvexHull): The computed convex hull object.

    Raises:
        Exception: If the number of points is less than n_features + 1.
    """

    m = X.shape[1]
    n = X.shape[0]
    if n < m + 1:
        raise Exception("must have nrow(X) >= ncol(X) + 1")

    # get midpoints of external (convex hull) facets and normal vectors
    qhull = ConvexHull(X, qhull_options="n")
    norms = np.zeros((qhull.simplices.shape[0], m))
    Xbound = np.zeros((qhull.simplices.shape[0], m))
    for i in range(qhull.simplices.shape[0]):
        Xbound[i,] = np.mean(X[qhull.simplices[i,], :], axis=0)
        norms[i,] = qhull.equations[i, 0:m]

    # norms off of the boundary points to get fringe candidates
    # p specifies distance between hull and boundary
    eps = np.sqrt(np.finfo(float).eps)
    alpha = np.zeros(Xbound.shape[0])

    # Initialize with infinity (for zero/small norms)
    ai = np.full([Xbound.shape[0], m], np.inf)

    # Positive norms (significant): distance to upper bound
    pos_mask = norms > eps
    ai[pos_mask] = (upper - Xbound[pos_mask]) / norms[pos_mask]

    # Negative norms (significant): distance to lower bound
    neg_mask = norms < -eps
    ai[neg_mask] = (lower - Xbound[neg_mask]) / norms[neg_mask]

    alpha = np.min(ai, axis=1)

    Xfringe = Xbound + norms * alpha[:, np.newaxis] * p

    return {"XF": Xfringe, "XB": Xbound, "qhull": qhull}


def tricands(
    X: np.ndarray,
    p: float = 0.5,
    fringe: bool = True,
    nmax: int = None,
    best: int = None,
    ordering: np.ndarray = None,
    vis: bool = False,
    imgname: str = "tricands.pdf",
    lower: float = 0,
    upper: float = 1,
) -> np.ndarray:
    """
    Generate Triangulation Candidates for Bayesian Optimization.
    Assumes a bounding box of [lower, upper]^m.

    Args:
        X (np.ndarray): Design matrix of shape (n_samples, n_features).
            Each row gives a design point and each column a feature.
        p (float, optional): Distance to the boundary for fringe candidates
            (0 = on hull, 1 = on boundary). Defaults to 0.5.
        fringe (bool, optional): Whether to include fringe points to allow
            exploration outside the convex hull. Defaults to True.
        nmax (int, optional): Maximum size of candidate set. If output exceeds this,
            strategic subsetting is employed. Defaults to 100 * n_features.
        best (int, optional): Index of the best (lowest) currently observed point.
            Used for strategic subsetting in Bayesian optimization.
            Defaults to None.
        ordering (np.ndarray, optional): Order of closeness of rows of X to a contour level.
            Used for contour location subsetting. Defaults to None.
        vis (bool, optional): Whether to visualize the triangulation.
            Only applicable to 2D designs. Defaults to False.
        imgname (str, optional): File name for saved plot if vis=True.
            Defaults to 'tricands.pdf'.
        lower (float, optional): Lower bound of bounding box for all dimensions. Defaults to 0.
        upper (float, optional): Upper bound of bounding box for all dimensions. Defaults to 1.

    Returns:
        np.ndarray: Array of candidate points, shape (n_candidates, n_features).

    Raises:
        Exception: If visualization is requested for non-2D data.
        Exception: If number of points is less than n_features + 1.
        Exception: If both 'best' and 'ordering' are provided.
        Exception: If X contains values outside [lower, upper].

    Examples:
        >>> import numpy as np
        >>> from spotoptim.tricands import tricands
        >>> X = np.array([[0.1, 0.1], [0.9, 0.1], [0.5, 0.9], [0.2, 0.5]])
        >>> candidates = tricands(X, fringe=True, p=0.5)
        >>> print(candidates.shape)
        (7, 2)
    """
    # extract dimsions and do sanity checks
    m = X.shape[1]
    n = X.shape[0]
    if nmax is None:
        nmax = 100 * m
    if vis and m != 2:
        raise Exception("visuals only possible when ncol(X) = 2")
    if n < m + 1:
        raise Exception("must have nrow(X) >= ncol(X) + 1")
    if best is not None and ordering is not None:
        raise Exception("can only subset for BO or CL, not both")
    if np.min(X) < lower or np.max(X) > upper:
        raise Exception("X outside of lower/upper bounds")

    # possible visual
    if vis:
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1])
        plt.xlim(lower, upper)
        plt.ylim(lower, upper)

    # interior candidates
    ic = tricands_interior(X)
    Xcand = ic["cand"]
    if vis:
        for i in range(ic["tri"].shape[0]):
            X[np.append(ic["tri"][i, :], ic["tri"][i, 0]),].T
            for j in range(ic["tri"].shape[1] + 1):
                if j < ic["tri"].shape[1]:
                    xpoints = X[ic["tri"][i, j : (j + 2)], 0]
                    ypoints = X[ic["tri"][i, j : (j + 2)], 1]
                else:
                    xpoints = X[ic["tri"][i, [-1, 0]], 0]
                    ypoints = X[ic["tri"][i, [-1, 0]], 1]
                plt.plot(xpoints, ypoints, color="black")

        plt.scatter(Xcand[:, 0], Xcand[:, 1])

    # calculate midpoints of convex hull vectors
    if fringe:
        fr = tricands_fringe(X, p, lower=lower, upper=upper)
        Xcand = np.concatenate([Xcand, fr["XF"]], axis=0)
        # possibly visualize fringe candidates
        if vis:
            for i in range(fr["XB"].shape[0]):
                plt.arrow(
                    fr["XB"][i, 0],
                    fr["XB"][i, 1],
                    fr["XF"][i, 0] - fr["XB"][i, 0],
                    fr["XF"][i, 1] - fr["XB"][i, 1],
                    width=0.005,
                    color="red",
                )

    # throw some away?
    if nmax < Xcand.shape[0]:

        adj = []

        # Bayesian optimization strategic subsetting
        if best is not None:
            # find candidates adjacent to best
            adj = np.where(
                np.apply_along_axis(lambda x: np.any(x == best), 1, ic["tri"])
            )[0]
            if len(adj) > nmax / 10:
                np.random.choice(adj, round(nmax / 10), replace=False)
            if vis:
                plt.scatter(
                    X[best : (best + 1), 0], X[best : (best + 1), 1], color="green"
                )
            if len(adj) >= nmax:
                raise Exception("adjacent to best >= nmax")

        # Contour location strategic subsetting
        if ordering is not None:
            i = 0
            facets = np.column_stack(
                (fr["qhull"].simplices, [None] * fr["qhull"].simplices.shape[0])
            )
            all_tri = np.vstack((ic["tri"], facets))
            while len(adj) < nmax / 10:
                # get all triangles adjacent to the i'th best point
                adj_tri = np.where(
                    np.apply_along_axis(lambda x: np.any(x == ordering[i]), 1, all_tri)
                )[0]
                if len(adj_tri) == 0 or i == len(ordering) - 1:
                    break
                else:
                    # remove duplicates
                    duplicates = [i in adj for i in adj_tri]
                    adj_tri = adj_tri[np.logical_not(duplicates)]
                    adj.append(np.random.choice(adj_tri))
                    i = i + 1
            # need to add optional visual

        # get the rest randomly
        remain = np.array(list(range(Xcand.shape[0])))
        if len(adj) > 0:
            remain = np.delete(remain, adj, 0)
        rest = np.random.choice(remain, (nmax - len(adj)), replace=False)
        sel = np.concatenate([adj, rest], axis=0).astype(int)
        Xcand = Xcand[sel, :]

        # possibly visualize
        if vis:
            plt.scatter(Xcand[:, 0], Xcand[:, 1], color="green")

    if vis:
        plt.savefig(imgname)
        plt.close()
    return Xcand
