from spotoptim.sampling.mm import mm_improvement, mmphi_intensive
from scipy.optimize import dual_annealing
import numpy as np
from typing import Tuple, List, Any


def mo_mm_desirability_function(
    x,
    models,
    X_base,
    J_base,
    d_base,
    phi_base,
    D_overall,
    mm_objective=True,
    verbose=False,
) -> Tuple[float, List[float]]:
    """
    Calculates the negative combined desirability for a candidate point x. Can be used by the mo_mm_desirability_optimizer.
    For each objective, a model is used to predict the objective value at x.
    If mm_objective is True, the Morris-Mitchell improvement is also calculated and included as an additional objective.
    The combined desirability, which uses the predictions from the models and optionally the Morris-Mitchell improvement,
    is then computed using the provided DOverall object.

    Args:
        x (np.ndarray):
            Candidate point (1D array).
        models (list):
            List of trained models. One model per objective.
        X_base (np.ndarray):
            Existing design points. Used for computing Morris-Mitchell improvement.
        J_base (np.ndarray):
            Multiplicities of distances for X_base. Used for Morris-Mitchell improvement.
        d_base (np.ndarray):
            Unique distances for X_base. Used for Morris-Mitchell improvement.
        phi_base (float):
            Base Morris-Mitchell metric for X_base. Used for Morris-Mitchell improvement.
        D_overall (DOverall):
            The overall desirability function. Must include desirability functions for each objective and optionally for Morris-Mitchell.
        mm_objective (bool):
            Whether to include space-filling improvement as an objective. Defaults to True.
        verbose (bool):
            Whether to print Morris-Mitchell improvement values. Defaults to False.

    Returns:
        Tuple[float, List[float]]:
            A tuple containing:
                - Negative geometric mean of desirabilities (for minimization).
                - List of individual objective values.

    Examples:
        >>> from spotoptim.mo import mo_mm_desirability_function
        >>> from spotdesirability import DOverall, DMax
        >>> import numpy as np
        >>> from spotoptim.function.mo import mo_conv2_max
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> from spotoptim.sampling.mm import mmphi_intensive
        >>> # X_base in the range [0,1]
        >>> X_base = np.random.rand(500, 2)
        >>> y = mo_conv2_max(X_base)
        >>> models = []
        >>> for i in range(y.shape[1]):
        ...     model = RandomForestRegressor(n_estimators=100, random_state=42)
        ...     model.fit(X_base, y[:, i])
        ...     models.append(model)
        >>> # calculate base Morris-Mitchell stats
        >>> phi_base, J_base, d_base = mmphi_intensive(X_base, q=2, p=2)
        >>> d_funcs = []
        >>> for i in range(y.shape[1]):
        ...     d_func = DMax(low=np.min(y[:, i]), high=np.max(y[:, i]))
        ...     d_funcs.append(d_func)
        >>> D_overall = DOverall(*d_funcs)
        >>> x_test = np.random.rand(2)  # Example test point
        >>> neg_D, objectives = mo_mm_desirability_function(x_test, models, X_base, J_base, d_base, phi_base, D_overall, mm_objective=False)
        >>> print(f"Negative Desirability: {neg_D}")
        Negative Desirability: ...
        >>> print(f"Objectives: {objectives}")
        Objectives: ...
    """
    # 1. Predict for all models
    x_reshaped = x.reshape(1, -1)
    predictions = [model.predict(x_reshaped)[0] for model in models]

    # 2. Compute y_mm (Space-filling improvement) if requested
    if mm_objective:
        y_mm = mm_improvement(x, X_base, phi_base, J_base, d_base, verbose=verbose)
        predictions.append(y_mm)

    # 3. Calculate combined desirability
    D = D_overall.predict(predictions)

    # Ensure D is a scalar
    if isinstance(D, np.ndarray):
        D = D.item()

    return -D, predictions


def mo_xy_desirability_plot(
    models: list,
    X_base: np.ndarray,
    J_base: np.ndarray,
    d_base: np.ndarray,
    phi_base: float,
    D_overall,
    bounds: list = None,
    mm_objective: bool = True,
    resolution: int = 50,
    feature_names: list = None,
    **kwargs: Any,
) -> None:
    """
    Generates a plot of the desirability landscape.
    Plots the 2-dim X values as points in the plane and colors them according to their desirability values.
    For each pair of inputs, x_i and x_j (with i < j), one plot is generated.

    Args:
        models (list):
            List of trained models (one per objective).
        X_base (np.ndarray):
            Existing design points.
        J_base (np.ndarray):
            Multiplicities of distances for X_base.
        d_base (np.ndarray):
            Unique distances for X_base.
        phi_base (float):
            Base Morris-Mitchell metric.
        D_overall (DOverall):
            The overall desirability function.
        bounds (list, optional):
            List of tuples (min, max) for each dimension. If None, derived from X_base.
        mm_objective (bool, optional):
            Whether to include space-filling improvement. Defaults to True.
        resolution (int, optional):
            Grid resolution for the plot. Defaults to 50.
        feature_names (list, optional):
            List of names for the input variables. Defaults to None.
        **kwargs:
            Additional arguments for plt.subplots (e.g., figsize).

    Returns:
        None

    Examples:
        >>> from spotoptim.mo.mo_mm import mo_xy_desirability_plot
        >>> import numpy as np
        >>> from spotoptim.function.mo import mo_conv2_max
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> from spotoptim.sampling.mm import mmphi_intensive
        >>> # X_base in the range [0,1]
        >>> X_base = np.random.rand(500, 2)
        >>> y = mo_conv2_max(X_base)
        >>> models = []
        >>> for i in range(y.shape[1]):
        ...     model = RandomForestRegressor(n_estimators=100, random_state=42)
        ...     model.fit(X_base, y[:, i])
        ...     models.append(model)
        >>> # calculate base Morris-Mitchell stats
        >>> phi_base, J_base, d_base = mmphi_intensive(X_base, q=2, p=2)
        >>> d_funcs = []
        >>> for i in range(y.shape[1]):
        ...     d_func = DMax(low=np.min(y[:, i]), high=np.max(y[:, i]))
        ...     d_funcs.append(d_func)
        >>> D_overall = DOverall(*d_funcs)
        >>> mo_xy_desirability_plot(models, X_base, J_base, d_base, phi_base, D_overall)
    """
    import itertools
    import matplotlib.pyplot as plt

    n_points, n_features = X_base.shape

    if bounds is None:
        bounds = [
            (np.min(X_base[:, i]), np.max(X_base[:, i])) for i in range(n_features)
        ]

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]

    # 1. Identify input pairs (i, j) with i < j
    feature_pairs = list(itertools.combinations(range(n_features), 2))
    n_pairs = len(feature_pairs)

    if n_pairs == 0:
        print("Need at least 2 features to generate plots.")
        return

    # 2. Setup Plot Grid
    # One plot per pair.
    # We can arrange them in a grid roughly square? Or just rows?
    # User example: "2 x 2 grid should be used".
    # Let's do a simple grid calculation.
    import math

    cols = int(math.ceil(math.sqrt(n_pairs)))
    rows = int(math.ceil(n_pairs / cols))

    if "figsize" not in kwargs:
        kwargs["figsize"] = (5 * cols, 4 * rows)

    fig, axes = plt.subplots(rows, cols, **kwargs)

    # Flatten axes for easy iteration
    if n_pairs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Base point for fixed dimensions (midpoint)
    x_mid = np.array([(b[0] + b[1]) / 2.0 for b in bounds])

    for idx, (i, j) in enumerate(feature_pairs):
        ax = axes[idx]

        # Create grid
        xi = np.linspace(bounds[i][0], bounds[i][1], resolution)
        xj = np.linspace(bounds[j][0], bounds[j][1], resolution)
        Xi, Xj = np.meshgrid(xi, xj)
        D_values = np.zeros(Xi.shape)

        # Compute Desirability
        for r in range(Xi.shape[0]):
            for c in range(Xi.shape[1]):
                # Construct candidate point
                x_point = x_mid.copy()
                x_point[i] = Xi[r, c]
                x_point[j] = Xj[r, c]

                neg_D, _ = mo_mm_desirability_function(
                    x_point,
                    models,
                    X_base,
                    J_base,
                    d_base,
                    phi_base,
                    D_overall,
                    mm_objective=mm_objective,
                )
                D_values[r, c] = -neg_D  # Store positive desirability

        # Plot
        contour = ax.contourf(Xi, Xj, D_values, levels=20, cmap="viridis")
        fig.colorbar(contour, ax=ax)
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[j])
        ax.set_title(f"Desirability (x{i} vs x{j})")

    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()


def mo_mm_desirability_optimizer(
    X_base, models, bounds, obj_func, **kwargs: Any
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Optimizes the multi-objective function to find the next best point.
    Returns the best point, its desirability, and the history of objective values.

    Args:
        X_base (np.ndarray):
            Existing design points.
        models (list):
            List of trained surrogate models for each objective.
        bounds (list):
            Bounds for each dimension.
        obj_func (callable):
            Objective function to compute desirability and objectives.
        **kwargs (Any):
            Additional arguments for the objective function.

    Returns:
        Tuple[np.ndarray, float, np.ndarray]:
            A tuple containing:
                - Best point (np.ndarray)
                - Best desirability (float)
                - History of objective values (np.ndarray)

    """
    # Pre-calculate base MM stats
    phi_base, J_base, d_base = mmphi_intensive(X_base, q=2, p=2)

    # List to store callback values
    callback_values = []

    # Define the objective wrapper
    def func(x):
        neg_D, objectives = obj_func(
            x, models, X_base, J_base, d_base, phi_base, **kwargs
        )
        callback_values.append(objectives)
        return neg_D

    # Run optimization
    result = dual_annealing(func, bounds=bounds, maxiter=100, seed=42)

    return result.x, -result.fun, np.array(callback_values)
