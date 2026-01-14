import numpy as np
from typing import Optional, List, Tuple

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    plt = None
    Axes3D = None


def plot_surrogate(
    optimizer: object,
    i: int = 0,
    j: int = 1,
    show: bool = True,
    alpha: float = 0.8,
    var_name: Optional[List[str]] = None,
    cmap: str = "jet",
    num: int = 100,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    add_points: bool = True,
    grid_visible: bool = True,
    contour_levels: int = 30,
    figsize: Tuple[int, int] = (12, 10),
) -> None:
    """Plot the surrogate model for two dimensions.

    Creates a 2x2 plot showing:
    - Top left: 3D surface of predictions
    - Top right: 3D surface of prediction uncertainty
    - Bottom left: Contour plot of predictions with evaluated points
    - Bottom right: Contour plot of prediction uncertainty

    Args:
        optimizer: SpotOptim instance containing optimization data and surrogate model.
        i (int, optional): Index of the first dimension to plot. Defaults to 0.
        j (int, optional): Index of the second dimension to plot. Defaults to 1.
        show (bool, optional): If True, displays the plot immediately. Defaults to True.
        alpha (float, optional): Transparency of the 3D surface plots (0=transparent, 1=opaque).
            Defaults to 0.8.
        var_name (list of str, optional): Names for each dimension. If None, uses instance var_name.
            Defaults to None.
        cmap (str, optional): Matplotlib colormap name. Defaults to 'jet'.
        num (int, optional): Number of grid points per dimension for mesh grid. Defaults to 100.
        vmin (float, optional): Minimum value for color scale. If None, determined from data.
            Defaults to None.
        vmax (float, optional): Maximum value for color scale. If None, determined from data.
            Defaults to None.
        add_points (bool, optional): If True, overlay evaluated points on contour plots.
            Defaults to True.
        grid_visible (bool, optional): If True, show grid lines on contour plots. Defaults to True.
        contour_levels (int, optional): Number of contour levels. Defaults to 30.
        figsize (tuple of int, optional): Figure size in inches (width, height). Defaults to (12, 10).

    Raises:
        ValueError: If optimization hasn't been run yet, or if i, j are invalid.
        ImportError: If matplotlib is not installed.

    Examples:
        >>> import numpy as np
        >>> from spotoptim import SpotOptim
        >>> from spotoptim.plot.visualization import plot_surrogate
        >>>
        >>> def sphere(X):
        ...     return np.sum(X**2, axis=1)
        >>>
        >>> # Initialize and run optimizer
        >>> opt = SpotOptim(fun=sphere, bounds=[(-5, 5), (-5, 5)],
        ...                 max_iter=20, n_initial=10, var_name=['x1', 'x2'])
        >>> result = opt.optimize()
        >>>
        >>> # Plot surrogate model for dimensions 0 and 1
        >>> plot_surrogate(opt, i=0, j=1, show=False)
    """
    if plt is None:
        raise ImportError(
            "matplotlib is required for plot_surrogate(). "
            "Install it with: pip install matplotlib"
        )

    # Validation
    if optimizer.X_ is None or optimizer.y_ is None:
        raise ValueError("No optimization data available. Run optimize() first.")

    k = optimizer.n_dim
    if i >= k or j >= k:
        raise ValueError(f"Dimensions i={i} and j={j} must be less than n_dim={k}.")
    if i == j:
        raise ValueError("Dimensions i and j must be different.")

    # Use instance var_name if not provided
    if var_name is None:
        var_name = optimizer.var_name

    # Generate mesh grid
    X_i, X_j, grid_points = optimizer._generate_mesh_grid(i, j, num)

    # Predict on grid
    y_pred, y_std = optimizer._predict_with_uncertainty(grid_points)
    Z_pred = y_pred.reshape(X_i.shape)
    Z_std = y_std.reshape(X_i.shape)

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Plot 1: 3D surface of predictions
    ax1 = fig.add_subplot(221, projection="3d")
    ax1.plot_surface(X_i, X_j, Z_pred, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
    ax1.set_title("Prediction Surface")
    ax1.set_xlabel(var_name[i] if var_name else f"x{i}")
    ax1.set_ylabel(var_name[j] if var_name else f"x{j}")
    ax1.set_zlabel("Prediction")

    # Plot 2: 3D surface of prediction uncertainty
    ax2 = fig.add_subplot(222, projection="3d")
    ax2.plot_surface(X_i, X_j, Z_std, cmap=cmap, alpha=alpha)
    ax2.set_title("Prediction Uncertainty Surface")
    ax2.set_xlabel(var_name[i] if var_name else f"x{i}")
    ax2.set_ylabel(var_name[j] if var_name else f"x{j}")
    ax2.set_zlabel("Std. Dev.")

    # Plot 3: Contour of predictions
    ax3 = fig.add_subplot(223)
    contour3 = ax3.contourf(
        X_i, X_j, Z_pred, levels=contour_levels, cmap=cmap, vmin=vmin, vmax=vmax
    )
    plt.colorbar(contour3, ax=ax3)
    if add_points:
        ax3.scatter(
            optimizer.X_[:, i],
            optimizer.X_[:, j],
            c="red",
            s=30,
            edgecolors="black",
            zorder=5,
            label="Evaluated points",
        )
        ax3.legend()
    ax3.set_title("Prediction Contour")
    ax3.set_xlabel(var_name[i] if var_name else f"x{i}")
    ax3.set_ylabel(var_name[j] if var_name else f"x{j}")
    ax3.grid(visible=grid_visible)

    # Plot 4: Contour of prediction uncertainty
    ax4 = fig.add_subplot(224)
    contour4 = ax4.contourf(X_i, X_j, Z_std, levels=contour_levels, cmap=cmap)
    plt.colorbar(contour4, ax=ax4)
    if add_points:
        ax4.scatter(
            optimizer.X_[:, i],
            optimizer.X_[:, j],
            c="red",
            s=30,
            edgecolors="black",
            zorder=5,
            label="Evaluated points",
        )
        ax4.legend()
    ax4.set_title("Uncertainty Contour")
    ax4.set_xlabel(var_name[i] if var_name else f"x{i}")
    ax4.set_ylabel(var_name[j] if var_name else f"x{j}")
    ax4.grid(visible=grid_visible)

    plt.tight_layout()

    if show:
        plt.show()


def plot_progress(
    optimizer: object,
    show: bool = True,
    log_y: bool = False,
    figsize: Tuple[int, int] = (10, 6),
    ylabel: str = "Objective Value",
    mo: bool = False,
) -> None:
    """Plot optimization progress showing all evaluations and best-so-far curve.

    This method visualizes the optimization history, displaying both individual
    function evaluations and the cumulative best value found. Initial design points
    are shown as individual scatter points with a light grey background region,
    while sequential optimization iterations are connected with lines.

    Args:
        optimizer: SpotOptim instance containing optimization data.
        show (bool, optional): Whether to display the plot. Defaults to True.
        log_y (bool, optional): Whether to use log scale for y-axis. Defaults to False.
        figsize (tuple, optional): Figure size as (width, height). Defaults to (10, 6).
        ylabel (str, optional): Label for y-axis. Defaults to "Objective Value".
        mo (bool, optional): Whether to plot individual objectives if available. Defaults to False.

    Raises:
        ValueError: If optimization hasn't been run yet.
        ImportError: If matplotlib is not installed.

    Examples:
        >>> import numpy as np
        >>> from spotoptim import SpotOptim
        >>> from spotoptim.plot.visualization import plot_progress
        >>>
        >>> def sphere(X):
        ...     return np.sum(X**2, axis=1)
        >>>
        >>> # Initialize and run optimizer
        >>> opt = SpotOptim(fun=sphere, bounds=[(-5, 5)]*2,
        ...                 max_iter=20, n_initial=10, seed=42)
        >>> result = opt.optimize()
        >>>
        >>> # Plot optimization progress (linear scale)
        >>> plot_progress(opt, log_y=False, show=False)
        >>>
        >>> # Plot with log scale
        >>> plot_progress(opt, log_y=True, show=False)
    """
    if plt is None:
        raise ImportError(
            "matplotlib is required for plot_progress(). "
            "Install it with: pip install matplotlib"
        )

    if optimizer.y_ is None or len(optimizer.y_) == 0:
        raise ValueError("No optimization data available. Run optimize() first.")

    history = optimizer.y_

    plt.figure(figsize=figsize)

    # Separate initial design points from sequential evaluations
    n_initial = min(optimizer.n_initial, len(history))
    initial_y = history[:n_initial]
    sequential_y = history[n_initial:]

    # Add light grey background for initial design region
    if n_initial > 0:
        plt.axvspan(0, n_initial, alpha=0.15, color="gray", zorder=0)

    # Plot multi-objective values if requested and available
    if mo and optimizer.y_mo is not None:
        n_samples, n_obj = optimizer.y_mo.shape
        x_all = np.arange(1, n_samples + 1)

        # Determine names
        names = optimizer.objective_names
        if names is None or len(names) != n_obj:
            names = [f"Objective {i+1}" for i in range(n_obj)]

        # Basic colors (excluding gray/red used for main plot)
        # Use a colormap or a set list
        _ = plt.cm.viridis(np.linspace(0, 1, n_obj))

        for i in range(n_obj):
            plt.plot(
                x_all,
                optimizer.y_mo[:, i],
                linestyle="--",
                marker="x",
                alpha=0.7,
                label=f"{names[i]}",
                zorder=1,
            )

    # Plot initial design points as scatter (not connected)
    if n_initial > 0:
        x_initial = np.arange(1, n_initial + 1)
        plt.scatter(
            x_initial,
            initial_y,
            alpha=0.6,
            s=50,
            label=f"Initial design (n={n_initial})",
            color="gray",
            edgecolors="black",
            linewidth=0.5,
            zorder=2,
        )

    # Plot sequential evaluations (connected with line)
    if len(sequential_y) > 0:
        x_sequential = np.arange(n_initial + 1, len(history) + 1)
        plt.plot(
            x_sequential,
            sequential_y,
            "o-",
            alpha=0.6,
            label="Sequential evaluations",
            markersize=5,
            zorder=3,
        )

    # Plot best-so-far curve starting after initial design
    if len(history) > n_initial:
        # Best so far across all evaluations
        best_so_far = np.minimum.accumulate(history)
        # Start the red line after initial design
        x_best = np.arange(n_initial + 1, len(history) + 1)
        y_best = best_so_far[n_initial:]
        plt.plot(
            x_best,
            y_best,
            "r-",
            linewidth=2,
            label="Best so far",
            zorder=4,
        )

    plt.xlabel("Iteration", fontsize=11)
    plt.ylabel(ylabel, fontsize=11)

    title = "Optimization Progress"
    if log_y:
        title += " (Log Scale)"
    plt.title(title, fontsize=12)

    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    if log_y:
        plt.yscale("log")

    plt.tight_layout()

    if show:
        plt.show()


def plot_important_hyperparameter_contour(
    optimizer: object,
    max_imp: int = 3,
    show: bool = True,
    alpha: float = 0.8,
    cmap: str = "jet",
    num: int = 100,
    add_points: bool = True,
    grid_visible: bool = True,
    contour_levels: int = 30,
    figsize: Tuple[int, int] = (12, 10),
) -> None:
    """Plot surrogate contours for all combinations of the top max_imp important parameters.

    This method identifies the most important parameters using importance scores,
    then generates surrogate contour plots for all pairwise combinations of these
    parameters. Factor (categorical) variables are handled by creating discrete grids
    and displaying factor level names on the axes.

    Args:
        optimizer: SpotOptim instance containing optimization data.
        max_imp (int, optional): Number of most important parameters to visualize.
            Defaults to 3. For max_imp=3, creates 3 plots: (0,1), (0,2), (1,2).
        show (bool, optional): If True, displays plots immediately. Defaults to True.
        alpha (float, optional): Transparency of 3D surface plots (0=transparent, 1=opaque).
            Defaults to 0.8.
        cmap (str, optional): Matplotlib colormap name. Defaults to 'jet'.
        num (int, optional): Number of grid points per dimension. Defaults to 100.
            For factor variables, uses the number of unique levels instead.
        add_points (bool, optional): If True, overlay evaluated points on contour plots.
            Defaults to True.
        grid_visible (bool, optional): If True, show grid lines. Defaults to True.
        contour_levels (int, optional): Number of contour levels. Defaults to 30.
        figsize (tuple of int, optional): Figure size in inches (width, height).
            Defaults to (12, 10).

    Raises:
        ValueError: If optimization hasn't been run yet or max_imp is invalid.

    Examples:
        >>> import numpy as np
        >>> from spotoptim import SpotOptim
        >>> from spotoptim.plot.visualization import plot_important_hyperparameter_contour
        >>>
        >>> def sphere(X):
        ...     return np.sum(X**2, axis=1)
        >>>
        >>> # Initialize and run optimizer with enough dimensions
        >>> opt = SpotOptim(fun=sphere, bounds=[(-5, 5)]*4,
        ...                 max_iter=20, n_initial=10,
        ...                 var_name=['x1', 'x2', 'x3', 'x4'])
        >>> result = opt.optimize()
        >>>
        >>> # Plot contours for top 3 important hyperparameters
        >>> plot_important_hyperparameter_contour(opt, max_imp=3, show=False)
    """
    from itertools import combinations

    if optimizer.X_ is None or optimizer.y_ is None:
        raise ValueError("No optimization data available. Run optimize() first.")

    if max_imp < 2:
        raise ValueError("max_imp must be at least 2 to generate pairwise plots.")

    if max_imp > optimizer.n_dim:
        raise ValueError(
            f"max_imp ({max_imp}) cannot exceed number of dimensions ({optimizer.n_dim})."
        )

    # Get importance scores
    importance = optimizer.get_importance()

    # Get indices of most important parameters (sorted by importance, descending)
    importance_array = np.array(importance)
    top_indices = np.argsort(importance_array)[::-1][:max_imp]

    # Get parameter names for informative output
    param_names = (
        optimizer.var_name
        if optimizer.var_name is not None
        else [f"x{i}" for i in range(len(importance))]
    )

    print(f"Plotting surrogate contours for top {max_imp} most important parameters:")
    for idx in top_indices:
        param_type = optimizer.var_type[idx] if optimizer.var_type else "float"
        print(
            f"  {param_names[idx]}: importance = {importance[idx]:.2f}% (type: {param_type})"
        )

    # Generate all pairwise combinations
    pairs = list(combinations(top_indices, 2))

    print(f"\nGenerating {len(pairs)} surrogate plots...")

    # Plot each combination
    for i, j in pairs:
        print(f"  Plotting {param_names[i]} vs {param_names[j]}")
        _plot_surrogate_with_factors(
            optimizer,
            i=int(i),
            j=int(j),
            show=show,
            alpha=alpha,
            cmap=cmap,
            num=num,
            add_points=add_points,
            grid_visible=grid_visible,
            contour_levels=contour_levels,
            figsize=figsize,
        )


def _plot_surrogate_with_factors(
    optimizer: object,
    i: int,
    j: int,
    show: bool = True,
    alpha: float = 0.8,
    cmap: str = "jet",
    num: int = 100,
    add_points: bool = True,
    grid_visible: bool = True,
    contour_levels: int = 30,
    figsize: Tuple[int, int] = (12, 10),
) -> None:
    """Plot surrogate model handling factor variables by mapping to integers.

    For factor variables, creates discrete grids and displays factor level names.

    Args:
        optimizer: SpotOptim instance.
        i (int): Index of the first dimension to plot.
        j (int): Index of the second dimension to plot.
        show (bool, optional): If True, displays the plot immediately. Defaults to True.
        alpha (float, optional): Transparency of the 3D surface plots (0=transparent, 1=opaque).
            Defaults to 0.8.
        cmap (str, optional): Matplotlib colormap name. Defaults to 'jet'.
        num (int, optional): Number of grid points per dimension for mesh grid. Defaults to 100.
        add_points (bool, optional): If True, overlay evaluated points on contour plots.
            Defaults to True.
        grid_visible (bool, optional): If True, show grid lines on contour plots. Defaults to True.
        contour_levels (int, optional): Number of contour levels. Defaults to 30.
        figsize (tuple of int, optional): Figure size in inches (width, height). Defaults to (12, 10).

    Raises:
        ImportError: If matplotlib is not installed.
    """
    if plt is None:
        raise ImportError(
            "matplotlib is required. Install with: pip install matplotlib"
        )

    # Check if either dimension is a factor
    is_factor_i = optimizer.var_type and optimizer.var_type[i] == "factor"
    is_factor_j = optimizer.var_type and optimizer.var_type[j] == "factor"

    # Get parameter names
    var_name = (
        optimizer.var_name
        if optimizer.var_name
        else [f"x{k}" for k in range(optimizer.n_dim)]
    )

    # Generate mesh grid with factor handling
    if is_factor_i or is_factor_j:
        (
            X_i,
            X_j,
            grid_points,
            factor_labels_i,
            factor_labels_j,
        ) = optimizer._generate_mesh_grid_with_factors(
            i, j, num, is_factor_i, is_factor_j
        )
    else:
        X_i, X_j, grid_points = optimizer._generate_mesh_grid(i, j, num)
        factor_labels_i, factor_labels_j = None, None

    # Predict on grid
    y_pred, y_std = optimizer._predict_with_uncertainty(grid_points)
    Z_pred = y_pred.reshape(X_i.shape)
    Z_std = y_std.reshape(X_i.shape)

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Plot 1: 3D surface of predictions
    ax1 = fig.add_subplot(221, projection="3d")
    ax1.plot_surface(X_i, X_j, Z_pred, cmap=cmap, alpha=alpha)
    ax1.set_title("Prediction Surface")
    ax1.set_xlabel(var_name[i] if var_name else f"x{i}")
    ax1.set_ylabel(var_name[j] if var_name else f"x{j}")
    ax1.set_zlabel("Prediction")

    # Handle factor labels on 3D plot ticks (simplified: often matplotlib 3D ticks are tricky, leaving defaults or using simple logic)
    if is_factor_i and factor_labels_i:
        ax1.set_xticks(range(len(factor_labels_i)))
        ax1.set_xticklabels(factor_labels_i, rotation=45, ha="right")
    if is_factor_j and factor_labels_j:
        ax1.set_yticks(range(len(factor_labels_j)))
        ax1.set_yticklabels(factor_labels_j, rotation=45, ha="right")

    # Plot 2: 3D surface of prediction uncertainty
    ax2 = fig.add_subplot(222, projection="3d")
    ax2.plot_surface(X_i, X_j, Z_std, cmap=cmap, alpha=alpha)
    ax2.set_title("Prediction Uncertainty Surface")
    ax2.set_xlabel(var_name[i] if var_name else f"x{i}")
    ax2.set_ylabel(var_name[j] if var_name else f"x{j}")
    ax2.set_zlabel("Std. Dev.")

    if is_factor_i and factor_labels_i:
        ax2.set_xticks(range(len(factor_labels_i)))
        ax2.set_xticklabels(factor_labels_i, rotation=45, ha="right")
    if is_factor_j and factor_labels_j:
        ax2.set_yticks(range(len(factor_labels_j)))
        ax2.set_yticklabels(factor_labels_j, rotation=45, ha="right")

    # Plot 3: Contour of predictions
    ax3 = fig.add_subplot(223)
    contour3 = ax3.contourf(X_i, X_j, Z_pred, levels=contour_levels, cmap=cmap)
    plt.colorbar(contour3, ax=ax3)

    if add_points:
        ax3.scatter(
            optimizer.X_[:, i],
            optimizer.X_[:, j],
            c="red",
            s=30,
            edgecolors="black",
            zorder=5,
            label="Evaluated points",
        )
        ax3.legend()
    ax3.set_title("Prediction Contour")
    ax3.set_xlabel(var_name[i] if var_name else f"x{i}")
    ax3.set_ylabel(var_name[j] if var_name else f"x{j}")

    if is_factor_i and factor_labels_i:
        ax3.set_xticks(range(len(factor_labels_i)))
        ax3.set_xticklabels(factor_labels_i, rotation=45, ha="right")
    if is_factor_j and factor_labels_j:
        ax3.set_yticks(range(len(factor_labels_j)))
        ax3.set_yticklabels(factor_labels_j, rotation=45, ha="right")

    ax3.grid(visible=grid_visible)

    # Plot 4: Contour of prediction uncertainty
    ax4 = fig.add_subplot(224)
    contour4 = ax4.contourf(X_i, X_j, Z_std, levels=contour_levels, cmap=cmap)
    plt.colorbar(contour4, ax=ax4)

    if add_points:
        ax4.scatter(
            optimizer.X_[:, i],
            optimizer.X_[:, j],
            c="red",
            s=30,
            edgecolors="black",
            zorder=5,
            label="Evaluated points",
        )
        ax4.legend()
    ax4.set_title("Uncertainty Contour")
    ax4.set_xlabel(var_name[i] if var_name else f"x{i}")
    ax4.set_ylabel(var_name[j] if var_name else f"x{j}")

    if is_factor_i and factor_labels_i:
        ax4.set_xticks(range(len(factor_labels_i)))
        ax4.set_xticklabels(factor_labels_i, rotation=45, ha="right")
    if is_factor_j and factor_labels_j:
        ax4.set_yticks(range(len(factor_labels_j)))
        ax4.set_yticklabels(factor_labels_j, rotation=45, ha="right")

    ax4.grid(visible=grid_visible)

    plt.tight_layout()

    if show:
        plt.show()
