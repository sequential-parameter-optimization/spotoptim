import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable


def simple_contour(
    fun,
    min_x=-1,
    max_x=1,
    min_y=-1,
    max_y=1,
    min_z=None,
    max_z=None,
    n_samples=100,
    n_levels=30,
) -> None:
    """
    Simple contour plot

    Args:
        fun (_type_): _description_
        min_x (int, optional): _description_. Defaults to -1.
        max_x (int, optional): _description_. Defaults to 1.
        min_y (int, optional): _description_. Defaults to -1.
        max_y (int, optional): _description_. Defaults to 1.
        min_z (int, optional): _description_. Defaults to 0.
        max_z (int, optional): _description_. Defaults to 1.
        n_samples (int, optional): _description_. Defaults to 100.
        n_levels (int, optional): _description_. Defaults to 5.

    Returns:
        None

    Examples:
        >>> import matplotlib.pyplot as plt
            import numpy as np
            # from spotpython.fun.objectivefunctions import analytical
            # fun = analytical().fun_branin
            # simple_contour(fun=fun, n_levels=30, min_x=-5, max_x=10, min_y=0, max_y=15)

    """
    XX, YY = np.meshgrid(
        np.linspace(min_x, max_x, n_samples), np.linspace(min_y, max_y, n_samples)
    )
    zz = np.array(
        [
            fun(np.array([xi, yi]).reshape(-1, 2))
            for xi, yi in zip(np.ravel(XX), np.ravel(YY))
        ]
    ).reshape(n_samples, n_samples)
    fig, ax = plt.subplots(figsize=(5, 2.7), layout="constrained")
    if min_z is None:
        min_z = np.min(zz)
    if max_z is None:
        max_z = np.max(zz)
    plt.contourf(
        XX,
        YY,
        zz,
        levels=np.linspace(min_z, max_z, n_levels),
        zorder=1,
        cmap="jet",
        vmin=min_z,
        vmax=max_z,
    )
    plt.colorbar()


def plotModel(
    model,
    lower,
    upper,
    i=0,
    j=1,
    min_z=None,
    max_z=None,
    var_type=None,
    var_name=None,
    show=True,
    filename=None,
    n_grid=50,
    contour_levels=10,
    dpi=200,
    title="",
    figsize=(12, 6),
    use_min=False,
    use_max=False,
    aspect_equal=True,
    legend_fontsize=12,
    cmap="viridis",
    X_points=None,
    y_points=None,
    plot_points=True,
    points_color="white",
    points_size=30,
    point_color_below="blue",
    point_color_above="red",
    atol=1e-6,
    plot_3d=False,
) -> None:
    """
    Generate 2D contour and optionally 3D surface plots for a model's predictions.

    Args:
        model (object): A model with a predict method.
        lower (array_like): Lower bounds for each dimension.
        upper (array_like): Upper bounds for each dimension.
        i (int): Index for the x-axis dimension.
        j (int): Index for the y-axis dimension.
        min_z (float, optional): Min value for color scaling. Defaults to None.
        max_z (float, optional): Max value for color scaling. Defaults to None.
        var_type (list, optional): Variable types for each dimension. Defaults to None.
        var_name (list, optional): Variable names for labeling axes. Defaults to None.
        show (bool): Whether to display the plot. Defaults to True.
        filename (str, optional): File path to save the figure. Defaults to None.
        n_grid (int): Resolution for each axis. Defaults to 50.
        contour_levels (int): Number of contour levels. Defaults to 10.
        dpi (int): DPI for saving. Defaults to 200.
        title (str): Title for the figure. Defaults to "".
        figsize (tuple): Figure size. Defaults to (12, 6).
        use_min (bool): If True, leftover dims are set to lower bounds.
        use_max (bool): If True, leftover dims are set to upper bounds.
        aspect_equal (bool): Whether axes have equal scaling. Defaults to True.
        legend_fontsize (int): Font size for labels and legends. Defaults to 12.
        cmap (str): Colormap. Defaults to "viridis".
        X_points (ndarray): Original data points. Shape: (N, D).
        y_points (ndarray): Original target values. Shape: (N,).
        plot_points (bool): Whether to plot X_points. Defaults to True.
        points_color (str): Fallback color for data points. Defaults to "white".
        points_size (int): Marker size for data points. Defaults to 30.
        point_color_below (str): Color if actual z < predicted z. Defaults to "blue".
        point_color_above (str): Color if actual z >= predicted z. Defaults to "red".
        atol (float): Absolute tolerance for comparing actual and predicted z-values. Defaults to 1e-6.
        plot_3d (bool): Whether to plot the 3D surface plot. Defaults to False.

    Returns:
        (fig, (ax_contour, ax_surface)): Figure and axes for the contour and surface plots.
    """
    # --- Validate inputs ---
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    n_dims = len(lower)
    if len(upper) != n_dims:
        raise ValueError("Mismatch in dimension count between lower and upper.")
    if i < 0 or j < 0 or i >= n_dims or j >= n_dims:
        raise ValueError(
            f"Invalid dimension indices i={i} or j={j} for {n_dims}-dimensional data."
        )
    if i == j:
        raise ValueError("Dimensions i and j must be different.")

    if var_name is None:
        var_name = [f"x{k}" for k in range(n_dims)]
    elif len(var_name) != n_dims:
        raise ValueError("var_name length must match the number of dimensions.")

    # --- 2D grid for contour/surface ---
    x_vals = np.linspace(lower[i], upper[i], n_grid)
    y_vals = np.linspace(lower[j], upper[j], n_grid)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

    # Helper for leftover dims
    def hidden_value(dim_index):
        if use_min:
            return lower[dim_index]
        if use_max:
            return upper[dim_index]
        return 0.5 * (lower[dim_index] + upper[dim_index])

    # Build all grid points
    grid_points = []
    for row in range(n_grid):
        for col in range(n_grid):
            p = np.zeros(n_dims)
            p[i] = X_grid[row, col]
            p[j] = Y_grid[row, col]
            for dim in range(n_dims):
                if dim not in (i, j):
                    p[dim] = hidden_value(dim)
            grid_points.append(p)
    grid_points = np.array(grid_points)

    # Predict for the grid
    Z_pred = model.predict(grid_points)
    if isinstance(Z_pred, dict):
        Z_pred = Z_pred.get("mean", list(Z_pred.values())[0])
    elif isinstance(Z_pred, tuple):
        Z_pred = Z_pred[0]
    Z_pred = Z_pred.reshape(n_grid, n_grid)

    # Determine min/max color scale
    if min_z is None:
        min_z = np.min(Z_pred)
    if max_z is None:
        max_z = np.max(Z_pred)

    # --- Set up figure ---
    if plot_3d:
        fig = plt.figure(figsize=figsize)
        ax_contour = fig.add_subplot(1, 2, 1)
        ax_surface = fig.add_subplot(1, 2, 2, projection="3d")
    else:
        fig, ax_contour = plt.subplots(figsize=figsize)
        ax_surface = None

    # --- 2D contour ---
    cont = ax_contour.contourf(
        X_grid,
        Y_grid,
        Z_pred,
        levels=contour_levels,
        cmap=cmap,
        vmin=min_z,
        vmax=max_z,
    )
    cb1 = plt.colorbar(cont, ax=ax_contour)
    cb1.ax.tick_params(labelsize=legend_fontsize - 2)

    ax_contour.set_xlabel(var_name[i], fontsize=legend_fontsize)
    ax_contour.set_ylabel(var_name[j], fontsize=legend_fontsize)
    ax_contour.tick_params(labelsize=legend_fontsize - 2)
    if aspect_equal:
        ax_contour.set_aspect("equal")

    # --- 3D surface ---
    if plot_3d:
        surf = ax_surface.plot_surface(
            X_grid,
            Y_grid,
            Z_pred,
            cmap=cmap,
            vmin=min_z,
            vmax=max_z,
            linewidth=0,
            antialiased=True,
            alpha=0.8,
        )
        cb2 = fig.colorbar(surf, ax=ax_surface, shrink=0.7, pad=0.1)
        cb2.ax.tick_params(labelsize=legend_fontsize - 2)

        ax_surface.set_xlabel(var_name[i], fontsize=legend_fontsize)
        ax_surface.set_ylabel(var_name[j], fontsize=legend_fontsize)
        ax_surface.set_zlabel("f(x)", fontsize=legend_fontsize)
        ax_surface.tick_params(labelsize=legend_fontsize - 2)

    # --- Optionally plot points ---
    if plot_points and X_points is not None:
        z_pred_for_point = []
        for row_idx in range(X_points.shape[0]):
            single_p = np.zeros(n_dims)
            single_p[i] = X_points[row_idx, i]
            single_p[j] = X_points[row_idx, j]
            for dim_idx in range(n_dims):
                if dim_idx not in (i, j):
                    single_p[dim_idx] = hidden_value(dim_idx)
            val = model.predict(single_p.reshape(1, -1))
            val = np.atleast_1d(val)
            if isinstance(val, dict):
                val = val.get("mean", list(val.values())[0])
            elif isinstance(val, tuple):
                val = val[0]
            z_pred_for_point.append(val[0] if hasattr(val, "__len__") else val)
        z_pred_for_point = np.array(z_pred_for_point)

        z_actual = np.array(y_points).flatten()

        on_mask = np.isclose(z_actual, z_pred_for_point, atol=atol)
        below_mask = z_actual - atol / 2.0 < z_pred_for_point
        above_mask = z_actual + atol / 2.0 > z_pred_for_point
        num_correct = np.count_nonzero(on_mask)

        # 2D contour scatter
        ax_contour.scatter(
            X_points[below_mask, i],
            X_points[below_mask, j],
            c=point_color_below,
            edgecolor="black",
            s=points_size,
            alpha=0.9,
            zorder=5,
        )
        ax_contour.scatter(
            X_points[above_mask, i],
            X_points[above_mask, j],
            c=point_color_above,
            edgecolor="black",
            s=points_size,
            alpha=0.9,
            zorder=5,
        )
        ax_contour.scatter(
            X_points[on_mask, i],
            X_points[on_mask, j],
            c=points_color,
            edgecolor="black",
            s=points_size,
            alpha=0.9,
            zorder=5,
        )
        # 3D plot scatter
        if plot_3d:
            ax_surface.scatter(
                X_points[below_mask, i],
                X_points[below_mask, j],
                z_actual[below_mask],
                c=point_color_below,
                edgecolor="black",
                s=points_size,
                alpha=0.9,
                zorder=10,
            )
            ax_surface.scatter(
                X_points[above_mask, i],
                X_points[above_mask, j],
                z_actual[above_mask],
                c=point_color_above,
                edgecolor="black",
                s=points_size,
                alpha=0.9,
                zorder=10,
            )
            ax_surface.scatter(
                X_points[on_mask, i],
                X_points[on_mask, j],
                z_actual[on_mask],
                c=points_color,
                edgecolor="black",
                s=points_size,
                alpha=0.9,
                zorder=10,
            )

    # --- Optionally set aspect in 3D ---
    if plot_3d and aspect_equal:
        x_range = upper[i] - lower[i]
        y_range = upper[j] - lower[j]
        z_range = max_z - min_z if max_z > min_z else 1
        scale_z = (x_range + y_range) / (2.0 * z_range) if z_range else 1
        ax_surface.set_box_aspect([1, (y_range / x_range) if x_range else 1, scale_z])

    # --- Title, save, and show ---
    if title:
        updated_title = f"{title}  Correct Points: {num_correct if plot_points and X_points is not None else ''}"
        fig.suptitle(updated_title, fontsize=legend_fontsize + 2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=dpi)

    if show:
        plt.show()

    if plot_3d:
        return fig, (ax_contour, ax_surface)
    else:
        return fig, (ax_contour, None)


def mo_generate_plot_grid(variables, resolutions, functions) -> pd.DataFrame:
    """
    Generate a grid of input variables and apply objective functions.

    Args:
        variables (dict): A dictionary where keys are variable names (e.g., "time", "temperature")
                          and values are tuples of (min_value, max_value).
        resolutions (dict): A dictionary where keys are variable names and values are the number of points.
        functions (dict): A dictionary where keys are function names and values are callable functions.

    Returns:
        pd.DataFrame: A DataFrame containing the grid and the results of the objective functions.
    """
    # Create a meshgrid for all variables
    grids = [
        np.linspace(variables[var][0], variables[var][1], resolutions[var])
        for var in variables
    ]
    grid = np.array(np.meshgrid(*grids)).T.reshape(-1, len(variables))

    # Create a DataFrame for the grid
    plot_grid = pd.DataFrame(grid, columns=variables.keys())

    input_cols = list(variables.keys())
    # Apply each function to the grid
    for func_name, func in functions.items():
        plot_grid[func_name] = plot_grid[input_cols].apply(
            lambda row: func(row.values), axis=1
        )

    return plot_grid


def contourf_plot(
    data,
    x_col,
    y_col,
    z_col,
    facet_col=None,
    aspect=1,
    as_table=True,
    figsize=(4, 4),
    levels=10,
    cmap="viridis",
    show_contour_lines=True,
    contour_line_color="black",
    contour_line_width=0.5,
    colorbar_orientation="vertical",
    wspace=0.4,
    hspace=0.4,
    highlight_point=None,
    highlight_color="red",
    highlight_marker="x",
    highlight_size=100,
) -> None:
    """
    Creates contour plots (single or faceted) using matplotlib.

    Args:
        data (pd.DataFrame): Data for plotting.
        x_col (str): Column name for x-axis.
        y_col (str): Column name for y-axis.
        z_col (str): Column name for z-axis (values).
        facet_col (str, optional): Column name for faceting.
        aspect (float, optional): Aspect ratio.
        as_table (bool, optional): If True, arranges facets in a table.
        figsize (tuple, optional): Figure size per plot key (if not faceted) or base.
        levels (int, optional): Number of contour levels.
        cmap (str, optional): Colormap.
        show_contour_lines (bool, optional): Whether to show contour lines.
        contour_line_color (str, optional): Color of contour lines.
        contour_line_width (float, optional): Width of contour lines.
        colorbar_orientation (str, optional): 'vertical' or 'horizontal'.
        wspace (float, optional): Width reserved for space between subplots.
        hspace (float, optional): Height reserved for space between subplots.
        highlight_point (dict-like, optional): Point to highlight. Must contain x_col, y_col keys.
                                               If faceted, must contain facet_col key.
        highlight_color (str, optional): Color of the highlight point.
        highlight_marker (str, optional): Marker style for the highlight point.
        highlight_size (int, optional): Size of the highlight point.
    """
    if facet_col:
        facet_values = data[facet_col].unique()
        num_facets = len(facet_values)

        # Determine subplot layout
        if as_table:
            num_cols = int(np.ceil(np.sqrt(num_facets)))
            num_rows = int(np.ceil(num_facets / num_cols))
        else:
            num_cols = num_facets
            num_rows = 1

        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(figsize[0] * num_cols, figsize[1] * num_rows),
            layout="constrained",
        )
        axes = np.array(axes).flatten()  # Flatten the axes array for easy indexing

        for i, facet_value in enumerate(facet_values):
            ax = axes[i]
            facet_data = data[data[facet_col] == facet_value]

            # Create grid for contour plot
            x = np.unique(facet_data[x_col])
            y = np.unique(facet_data[y_col])
            X, Y = np.meshgrid(x, y)
            Z = facet_data.pivot_table(index=y_col, columns=x_col, values=z_col).values

            # Plot contour
            contour = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
            if show_contour_lines:
                ax.contour(
                    X,
                    Y,
                    Z,
                    levels=levels,
                    colors=contour_line_color,
                    linewidths=contour_line_width,
                )

            # Highlight point if it belongs to this facet
            if highlight_point is not None:
                # Check if highlight_point belongs to current facet
                # Use approximate comparison for floats
                hp_facet_val = highlight_point.get(facet_col)
                if hp_facet_val is not None and np.isclose(hp_facet_val, facet_value):
                    ax.scatter(
                        highlight_point[x_col],
                        highlight_point[y_col],
                        color=highlight_color,
                        marker=highlight_marker,
                        s=highlight_size,
                        zorder=10,
                        label="Best",
                    )

            # Set labels and title
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{facet_col} = {np.round(facet_value, 2)}")
            ax.set_aspect(aspect)

        # Remove empty subplots
        for i in range(num_facets, len(axes)):
            fig.delaxes(axes[i])

        # Add colorbar
        fig.colorbar(
            contour,
            ax=axes.ravel().tolist(),
            orientation=colorbar_orientation,
            shrink=0.8,
            pad=0.05,
        )

        # fig.subplots_adjust(wspace=wspace, hspace=hspace)  # Incompatible with constrained_layout
        plt.show()

    else:
        # Create grid for contour plot
        x = np.unique(data[x_col])
        y = np.unique(data[y_col])
        X, Y = np.meshgrid(x, y)
        Z = data.pivot_table(index=y_col, columns=x_col, values=z_col).values

        # Plot contour
        fig, ax = plt.subplots(figsize=figsize)
        contour = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
        if show_contour_lines:
            ax.contour(
                X,
                Y,
                Z,
                levels=levels,
                colors=contour_line_color,
                linewidths=contour_line_width,
            )

        # Highlight point
        if highlight_point is not None:
            ax.scatter(
                highlight_point[x_col],
                highlight_point[y_col],
                color=highlight_color,
                marker=highlight_marker,
                s=highlight_size,
                zorder=10,
                label="Best",
            )

        # Set labels and title
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Contour Plot of {z_col}")
        ax.set_aspect(aspect)

        # Add colorbar using make_axes_locatable to ensure it's outside
        divider = make_axes_locatable(ax)
        if colorbar_orientation == "vertical":
            cax = divider.append_axes("right", size="5%", pad=0.1)
        else:
            cax = divider.append_axes("bottom", size="5%", pad=0.1)

        fig.colorbar(
            contour,
            cax=cax,
            orientation=colorbar_orientation,
        )

        plt.show()
