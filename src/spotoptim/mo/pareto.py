import numpy as np
from typing import Any
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def is_pareto_efficient(costs: np.ndarray, minimize: bool = True) -> np.ndarray:
    """
    Find the Pareto-efficient points from a set of points.

    A point is Pareto-efficient if no other point exists that is better in all objectives.
    This function assumes that lower values are preferred for each objective when `minimize=True`,
    and higher values are preferred when `minimize=False`.

    Args:
        costs (np.ndarray):
            An (N,M) array-like object of points, where N is the number of points and M is the number of objectives.
        minimize (bool, optional):
            If True, the function finds Pareto-efficient points assuming
            lower values are better. If False, it assumes higher values are better.
            Defaults to True.

    Returns:
        np.ndarray:
            A boolean mask of length N, where True indicates that the corresponding point is Pareto-efficient.

    Examples:

        >>> import numpy as np
        >>> from spotoptim.mo.pareto import is_pareto_efficient
        >>> points = np.array([[1, 2], [2, 1], [1.5, 1.5], [3, 3]])
        >>> pareto_mask = is_pareto_efficient(points, minimize=True)
        >>> print(pareto_mask)
        [ True  True  True False]
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, cost in enumerate(costs):
        if is_efficient[i]:
            if minimize:
                is_efficient[is_efficient] = np.any(costs[is_efficient] < cost, axis=1)
            else:
                is_efficient[is_efficient] = np.any(costs[is_efficient] > cost, axis=1)
            is_efficient[i] = True
    return is_efficient


def _get_mo_plot_data(
    model, bounds, i, j, resolution, x_base
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper to generate grid and predictions for a pair of features."""
    # Create grid for x_i, x_j
    xi_vals = np.linspace(bounds[i][0], bounds[i][1], resolution)
    xj_vals = np.linspace(bounds[j][0], bounds[j][1], resolution)
    Xi, Xj = np.meshgrid(xi_vals, xj_vals)

    # Prepare input array
    # We need to flatten Xi, Xj and repeat x_base for other dimensions
    n_points = resolution * resolution
    X_grid = np.tile(x_base, (n_points, 1))

    X_grid[:, i] = Xi.ravel()
    X_grid[:, j] = Xj.ravel()

    # Predict
    try:
        y_pred = model.predict(X_grid)
    except ValueError as e:
        # Catch shape mismatch errors typical of sklearn models
        if "features" in str(e) and "expecting" in str(e):
            raise ValueError(
                f"Model evaluation failed. This often happens if 'bounds' has a different length "
                f"({len(bounds)}) than the number of features the model expects. "
                "Please ensure len(bounds) matches the model's input dimensionality.\n"
                f"Original error: {e}"
            ) from e
        raise e

    # Reshape for plotting
    Z = y_pred.reshape(Xi.shape)

    return Xi, Xj, Z


def mo_xy_surface(
    models: list,
    bounds: list,
    target_names: list = None,
    feature_names: list = None,
    resolution: int = 50,
    feature_pairs: list = None,
    **kwargs: Any,
) -> None:
    """
    Generates surface plots of every combination of two input variables x_i and x_j
    (where i < j) and for each of the multiple objectives f_k.

    Args:
        models (list):
            List of trained models (one per objective).
        bounds (list):
            List of tuples (min, max) for each input variable.
        target_names (list, optional):
            List of names for the objectives. Defaults to None.
        feature_names (list, optional):
            List of names for the input variables. Defaults to None.
        resolution (int, optional):
            Grid resolution for the surface plot. Defaults to 50.
        feature_pairs (list, optional):
            List of tuples (i, j) specifying which feature pairs to plot.
            If None, all combinations are plotted. Defaults to None.
        **kwargs (Any):
            Additional keyword arguments passed to plt.subplots (e.g., figsize).

    Returns:
        None

    Examples:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> from spotoptim.mo.pareto import mo_xy_surface
        >>> import numpy as np
        >>> # Train dummy models
        >>> X = np.random.rand(10, 2)
        >>> y1 = X[:, 0] + X[:, 1]
        >>> y2 = X[:, 0] * X[:, 1]
        >>> m1 = RandomForestRegressor().fit(X, y1)
        >>> m2 = RandomForestRegressor().fit(X, y2)
        >>> # Plot
        >>> mo_xy_surface([m1, m2], bounds=[(0, 1), (0, 1)], target_names=["Sum", "Prod"])
    """
    # Validate bounds
    for i, b in enumerate(bounds):
        if not (np.isscalar(b[0]) and np.isscalar(b[1])):
            raise ValueError(
                f"Bounds for feature {i} must be scalars, but got {b}. "
                "Please ensure that bounds are a list of (min, max) tuples with scalar values."
            )

    n_features = len(bounds)
    n_objectives = len(models)

    # Check if models have n_features_in_ (sklearn API) and validate against bounds
    for m_idx, model in enumerate(models):
        if hasattr(model, "n_features_in_") and model.n_features_in_ != n_features:
            raise ValueError(
                f"Model {m_idx} expects {model.n_features_in_} features, but {n_features} bounds were provided. "
                "The number of bounds must match the number of features the model was trained on."
            )

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]
    if target_names is None:
        target_names = [f"f{i}" for i in range(n_objectives)]

    # Generate all pairs of features (i, j) with i < j
    if feature_pairs is None:
        feature_pairs = list(itertools.combinations(range(n_features), 2))
    n_pairs = len(feature_pairs)

    if n_pairs == 0:
        print("Need at least 2 features to generate surface plots.")
        return

    # Validate feature pairs
    for i, j in feature_pairs:
        if not (0 <= i < n_features and 0 <= j < n_features):
            raise ValueError(
                f"Invalid feature pair ({i}, {j}). Indices must be between 0 and {n_features-1}."
            )

    # Create a grid of subplots
    rows = n_pairs
    cols = n_objectives

    # Handle figsize in kwargs or default
    if "figsize" not in kwargs:
        kwargs["figsize"] = (5 * cols, 4 * rows)

    fig = plt.figure(**kwargs)

    # Base point (midpoint of bounds)
    x_base = np.array([(b[0] + b[1]) / 2.0 for b in bounds])

    plot_idx = 1
    for i, j in feature_pairs:
        for k, model in enumerate(models):
            ax = fig.add_subplot(rows, cols, plot_idx, projection="3d")

            Xi, Xj, Z = _get_mo_plot_data(model, bounds, i, j, resolution, x_base)

            # Plot surface
            ax.plot_surface(Xi, Xj, Z, cmap="viridis", edgecolor="none", alpha=0.8)

            ax.set_xlabel(feature_names[i])
            ax.set_ylabel(feature_names[j])
            ax.set_title(f"{target_names[k]}")

            plot_idx += 1

    plt.tight_layout()
    plt.show()


def mo_xy_contour(
    models: list,
    bounds: list,
    target_names: list = None,
    feature_names: list = None,
    resolution: int = 50,
    feature_pairs: list = None,
    **kwargs: Any,
) -> None:
    """
    Generates contour plots of every combination of two input variables x_i and x_j
    (where i < j) and for each of the multiple objectives f_k.

    Args:
        models (list):
            List of trained models (one per objective).
        bounds (list):
            List of tuples (min, max) for each input variable.
        target_names (list, optional):
            List of names for the objectives. Defaults to None.
        feature_names (list, optional):
            List of names for the input variables. Defaults to None.
        resolution (int, optional):
            Grid resolution for the contour plot. Defaults to 50.
        feature_pairs (list, optional):
            List of tuples (i, j) specifying which feature pairs to plot.
            If None, all combinations are plotted. Defaults to None.
        **kwargs (Any):
            Additional keyword arguments passed to plt.subplots (e.g., figsize).

    Returns:
        None

    Examples:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> from spotoptim.mo.pareto import mo_xy_contour
        >>> import numpy as np
        >>> # Train dummy models
        >>> X = np.random.rand(10, 2)
        >>> y1 = X[:, 0] + X[:, 1]
        >>> y2 = X[:, 0] * X[:, 1]
        >>> m1 = RandomForestRegressor().fit(X, y1)
        >>> m2 = RandomForestRegressor().fit(X, y2)
        >>> # Plot
        >>> mo_xy_contour([m1, m2], bounds=[(0, 1), (0, 1)], target_names=["Sum", "Prod"])
    """
    # Validate bounds
    for i, b in enumerate(bounds):
        if not (np.isscalar(b[0]) and np.isscalar(b[1])):
            raise ValueError(
                f"Bounds for feature {i} must be scalars, but got {b}. "
                "Please ensure that bounds are a list of (min, max) tuples with scalar values."
            )

    n_features = len(bounds)
    n_objectives = len(models)

    # Check if models have n_features_in_ (sklearn API) and validate against bounds
    for m_idx, model in enumerate(models):
        if hasattr(model, "n_features_in_") and model.n_features_in_ != n_features:
            raise ValueError(
                f"Model {m_idx} expects {model.n_features_in_} features, but {n_features} bounds were provided. "
                "The number of bounds must match the number of features the model was trained on."
            )

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]
    if target_names is None:
        target_names = [f"f{i}" for i in range(n_objectives)]

    # Generate all pairs of features (i, j) with i < j
    if feature_pairs is None:
        feature_pairs = list(itertools.combinations(range(n_features), 2))
    n_pairs = len(feature_pairs)

    if n_pairs == 0:
        print("Need at least 2 features to generate contour plots.")
        return

    # Validate feature pairs
    for i, j in feature_pairs:
        if not (0 <= i < n_features and 0 <= j < n_features):
            raise ValueError(
                f"Invalid feature pair ({i}, {j}). Indices must be between 0 and {n_features-1}."
            )

    # Create a grid of subplots
    rows = n_pairs
    cols = n_objectives

    # Handle figsize in kwargs or default
    if "figsize" not in kwargs:
        kwargs["figsize"] = (5 * cols, 4 * rows)

    fig, axes = plt.subplots(rows, cols, **kwargs)

    # Ensure axes is always iterable (handle 1x1, 1xN, Nx1 cases)
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Base point (midpoint of bounds)
    x_base = np.array([(b[0] + b[1]) / 2.0 for b in bounds])

    plot_idx = 0
    for i, j in feature_pairs:
        for k, model in enumerate(models):
            ax = axes[plot_idx]

            Xi, Xj, Z = _get_mo_plot_data(model, bounds, i, j, resolution, x_base)

            # Plot contour
            cp = ax.contourf(Xi, Xj, Z, cmap="viridis", alpha=0.8, levels=20)
            fig.colorbar(cp, ax=ax)  # Add colorbar to each plot

            ax.set_xlabel(feature_names[i])
            ax.set_ylabel(feature_names[j])
            ax.set_title(f"{target_names[k]}")

            plot_idx += 1

    plt.tight_layout()
    plt.show()


def mo_pareto_optx_plot(
    X: np.ndarray,
    Y: np.ndarray,
    minimize: bool = True,
    feature_names: list = None,
    target_names: list = None,
    **kwargs: Any,
) -> None:
    """
    Visualizes the Pareto-optimal points in the input space for each pair of inputs
    x_i and x_j (with i < j) and each objective f_k.

    Plots are placed on a grid where rows correspond to input pairs and columns
    correspond to objectives.

    Args:
        X (np.ndarray):
            An (N,D) array of input points, where N is the number of points and
            D is the number of variables (dimensions).
        Y (np.ndarray):
            An (N,M) array of objective values, where N is the number of points and
            M is the number of objectives.
        minimize (bool, optional):
            If True, assumes minimization of objectives. Defaults to True.
        feature_names (list, optional):
            List of names for the input variables. Defaults to None.
        target_names (list, optional):
            List of names for the objectives. Defaults to None.
        **kwargs (Any):
            Additional arguments passed to plt.subplots (e.g., figsize).

    Returns:
        None

    Examples:
        >>> from spotoptim.mo.pareto import mo_pareto_optx_plot
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> Y = np.array([[1, 2], [3, 4], [5, 6]])
        >>> mo_pareto_optx_plot(X, Y)
    """
    n_points, n_features = X.shape
    _, n_objectives = Y.shape

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]
    if target_names is None:
        target_names = [f"f{i}" for i in range(n_objectives)]

    # 1. Determine Pareto-efficient points
    pareto_mask = is_pareto_efficient(Y, minimize=minimize)

    # 2. Identify input pairs (i, j) with i < j
    feature_pairs = list(itertools.combinations(range(n_features), 2))
    n_pairs = len(feature_pairs)

    if n_pairs == 0:
        print("Need at least 2 features to generate plots.")
        return

    # 3. Create grid of plots
    # Grid: Rows = Feature Pairs, Cols = Objectives
    rows = n_pairs
    cols = n_objectives

    # Handle figsize in kwargs or default
    if "figsize" not in kwargs:
        # Default size estimation: 4 inches per row, 5 inches per col
        kwargs["figsize"] = (5 * cols, 4 * rows)

    fig, axes = plt.subplots(rows, cols, **kwargs)

    # Ensure axes is always a 2D array for consistent indexing
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # 4. Iterate and plot
    for r, (i, j) in enumerate(feature_pairs):
        for c in range(n_objectives):
            ax = axes[r, c]

            # Scatter all points (not pareto) - maybe faint?
            # User request: "visualize the Pareto-optimal points"
            # "points in the input space that correspond to the Pareto front"
            # Let's plot all points first for context, then highlight Pareto.

            # Plot non-Pareto points
            ax.scatter(
                X[~pareto_mask, i],
                X[~pareto_mask, j],
                c="lightgray",
                label="Non-Pareto",
                alpha=0.5,
                s=20,
            )

            # Plot Pareto points
            # Color by objective value f_k (Y[:, c])
            sc = ax.scatter(
                X[pareto_mask, i],
                X[pareto_mask, j],
                c=Y[pareto_mask, c],
                cmap="viridis",
                label="Pareto",
                s=50,
                edgecolor="k",
            )

            ax.set_xlabel(feature_names[i])
            ax.set_ylabel(feature_names[j])
            ax.set_title(f"{target_names[c]} (Color)")

            # Add colorbar for this subplot
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label(target_names[c])

    plt.tight_layout()
    plt.show()
