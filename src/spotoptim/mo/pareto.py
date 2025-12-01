import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def plot_mo(
    combinations: list,
    pareto: str,
    y_rf: np.ndarray = None,
    pareto_front_y_rf: bool = False,
    y_best: np.ndarray = None,
    y_orig: np.ndarray = None,
    pareto_front_orig: bool = False,
    pareto_label: bool = False,
    target_names: list = None,
    y_rf_color="blue",
    y_best_color="red",
    title: str = "",
) -> None:
    """
    Generates scatter plots for each combination of two targets from a multi-output
    prediction while highlighting Pareto optimal points.

    Args:
        combinations (list):
            A list of tuples, where each tuple contains two indices representing
            the targets to be plotted against each other.
        pareto (str):
            A string indicating whether to minimize ('min') or maximize ('max') the objectives.
        y_rf (np.ndarray, optional):
            An (N,M) array-like object of predicted values from the model, where N is the number of points and M is the number of objectives.
            Defaults to None.
        pareto_front_y_rf (bool, optional):
            If True, draws the Pareto front for the predicted values. Defaults to False.
        y_best (np.ndarray, optional):
            An (1,M) array-like object representing the best observed values. Defaults to None.
        y_orig (np.ndarray, optional):
            An (N,M) array-like object of original observed values, where N is the number of points and M is the number of objectives.
            Defaults to None.
        pareto_front_orig (bool, optional):
            If True, draws the Pareto front for the original observed values. Defaults to False.
        pareto_label (bool, optional):
            If True, labels the Pareto optimal points with their indices. Defaults to False.
        target_names (list, optional):
            A list of strings representing the names of the targets for labeling axes. Defaults to None.
        y_rf_color (str, optional):
            Color for the predicted points and Pareto front. Defaults to "blue".
        y_best_color (str, optional):
            Color for the best observed point. Defaults to "red".
        title (str, optional):
            Title for the plots. Defaults to an empty string.

    Returns:
        None

    Examples:
        >>> import numpy as np
        >>> from spotoptim.mo.pareto import plot_mo
        >>> y_rf = np.array([[1, 2], [2, 1], [1.5, 1.5], [3, 3]])
        >>> y_orig = np.array([[1.2, 2.1], [2.2, 0.9], [1.4, 1.6], [2.9, 3.1]])
        >>> y_best = np.array([1, 1])
        >>> combinations = [(0, 1)]
        >>> plot_mo(
        ...     combinations=combinations,
        ...     pareto='min',
        ...     y_rf=y_rf,
        ...     y_orig=y_orig,
        ...     y_best=y_best,
        ...     pareto_front_y_rf=True,
        ...     pareto_front_orig=True,
        ...     pareto_label=True,
        ...     target_names=['Objective 1', 'Objective 2'],
        ...     title='Multi-Objective Optimization Results'
        ... )
    """
    # Convert y_rf to numpy array if it's a pandas DataFrame
    if isinstance(y_rf, pd.DataFrame):
        y_rf = y_rf.values

    # Convert y_orig to numpy array if it's a pandas DataFrame
    if isinstance(y_orig, pd.DataFrame):
        y_orig = y_orig.values

    for i, j in combinations:
        plt.figure(figsize=(8, 6))
        s = 50  # Base size for points
        pareto_size = s  # Size for Pareto points
        if pareto_label:
            pareto_size = s * 4  # Increase the size for Pareto points
        a = 0.4

        # Plot original data if provided
        if y_orig is not None:
            # Determine Pareto optimal points for original data
            minimize = pareto == "min"
            pareto_mask_orig = is_pareto_efficient(y_orig[:, [i, j]], minimize)

            # Plot all original points
            plt.scatter(
                y_orig[:, i],
                y_orig[:, j],
                edgecolor="w",
                c="gray",
                s=s,
                marker="o",
                alpha=a,
                label="Original Points",
            )

            # Highlight Pareto points for original data
            plt.scatter(
                y_orig[pareto_mask_orig, i],
                y_orig[pareto_mask_orig, j],
                edgecolor="k",
                c="gray",
                s=pareto_size,
                marker="o",
                alpha=a,
                label="Original Pareto",
            )

            # Label Pareto points for original data if requested
            if pareto_label:
                for idx in np.where(pareto_mask_orig)[0]:
                    plt.text(
                        y_orig[idx, i],
                        y_orig[idx, j],
                        str(idx),
                        color="black",
                        fontsize=8,
                        ha="center",
                        va="center",
                    )

            # Draw Pareto front for original data if requested
            if pareto_front_orig:
                sorted_indices_orig = np.argsort(y_orig[pareto_mask_orig, i])
                plt.plot(
                    y_orig[pareto_mask_orig, i][sorted_indices_orig],
                    y_orig[pareto_mask_orig, j][sorted_indices_orig],
                    "k-",
                    alpha=a,
                    label="Original Pareto Front",
                )

        if y_rf is not None:
            # Determine Pareto optimal points for predicted data
            minimize = pareto == "min"
            pareto_mask = is_pareto_efficient(y_rf[:, [i, j]], minimize)

            # Plot all predicted points
            plt.scatter(
                y_rf[:, i],
                y_rf[:, j],
                edgecolor="w",
                c=y_rf_color,
                s=s,
                marker="^",
                alpha=a,
                label="Predicted Points",
            )

            # Highlight Pareto points for predicted data
            plt.scatter(
                y_rf[pareto_mask, i],
                y_rf[pareto_mask, j],
                edgecolor="k",
                c=y_rf_color,
                s=pareto_size,
                marker="s",
                alpha=a,
                label="Predicted Pareto",
            )

            # Label Pareto points for predicted data if requested
            if pareto_label:
                for idx in np.where(pareto_mask)[0]:
                    plt.text(
                        y_rf[idx, i],
                        y_rf[idx, j],
                        str(idx),
                        color="black",
                        fontsize=8,
                        ha="center",
                        va="center",
                    )

            # Draw Pareto front for predicted data if requested
            if pareto_front_y_rf:
                sorted_indices = np.argsort(y_rf[pareto_mask, i])
                plt.plot(
                    y_rf[pareto_mask, i][sorted_indices],
                    y_rf[pareto_mask, j][sorted_indices],
                    linestyle="-",  # Specify the line style
                    color=y_rf_color,  # Use the color specified by y_rf_color
                    alpha=a,
                    label="Predicted Pareto Front",
                )

        # Plot the best point, if provided
        if y_best is not None:
            # Ensure y_best is 2D
            if y_best.ndim == 1:
                y_best = y_best.reshape(1, -1)
            plt.scatter(
                y_best[:, i],
                y_best[:, j],
                edgecolor="k",
                c=y_best_color,
                s=s,
                marker="D",
                alpha=1,
                label="Best",
            )

        if target_names is not None:
            plt.xlabel(target_names[i])
            plt.ylabel(target_names[j])
        plt.grid()
        plt.title(title)
        plt.legend()
        plt.show()
