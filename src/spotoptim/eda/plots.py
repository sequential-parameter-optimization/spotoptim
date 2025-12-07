import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spotoptim.utils.stats import calculate_outliers


def plot_ip_histograms(
    df: pd.DataFrame,
    bins=10,
    num_cols=2,
    figwidth=10,
    thrs_unique=5,
    add_points: pd.DataFrame = None,
    add_points_col: list = ["red"],
) -> None:
    """
    Generate infill-point histograms (ip-histograms) for each numerical column in the DataFrame within a single figure.
    The title of each histogram shows the total, unique values count, outliers, and standard deviation.
    If there are fewer unique values than the threshold thrs_unique, the ip-histogram is colored differently.
    Additional points can be added and highlighted in red.

    Args:
        df (pd.DataFrame): DataFrame containing the data to plot.
        bins (int, optional): Number of bins for the histograms. Defaults to 10.
        num_cols (int, optional): Number of columns in the subplot grid. Defaults to 2.
        figwidth (int, optional): Width of the entire figure. Defaults to 10.
        thrs_unique (int, optional): Threshold for unique values to change histogram color. Defaults to 5.
        add_points (pd.DataFrame, optional): DataFrame containing additional points to highlight. Defaults to None.
        add_points_col (list, optional): List of colors for the additional points. Defaults to ["red"].

    Returns:
        None

    Examples:
        >>> import pandas as pd
        >>> from spotoptim.eda.plots import plot_ip_histograms
        >>> data = {'A': [1, 2, 2, 3, 4, 5, 100], 'B': [10, 10, 10, 10, 10, 10, 10]}
        >>> df = pd.DataFrame(data)
        >>> plot_ip_histograms(df, bins=5, num_cols=1, thrs_unique=3)
        >>> # Example with multiple added points and colors
        >>> add_points = pd.DataFrame({'A': [1.5, 3.5], 'B': [10, 10]})
        >>> plot_ip_histograms(df, add_points=add_points, add_points_col=["red", "blue"])
    """
    numerical_columns = df.select_dtypes(include="number").columns.tolist()
    num_plots = len(numerical_columns)
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=(figwidth, num_rows * 5)
    )
    # Ensure axes is always an array
    if num_rows == 1 and num_cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    for i, col in enumerate(numerical_columns):
        ax = axes[i]
        data = df[col].dropna()
        total_points = data.size
        unique_values = data.nunique()
        num_outliers = calculate_outliers(data)
        std_dev = data.std()
        fill_color = "lightcoral" if unique_values < thrs_unique else "lightblue"
        ax.hist(data, bins=bins, alpha=0.7, color=fill_color, edgecolor="black")
        if add_points is not None and col in add_points.columns:
            if len(add_points) != len(add_points_col):
                raise ValueError(
                    f"Length of add_points ({len(add_points)}) and add_points_col ({len(add_points_col)}) must be the same."
                )

            points_data = add_points[[col]].copy()
            points_data["color"] = add_points_col
            points_data = points_data.dropna(subset=[col])

            points = points_data[col]
            colors = points_data["color"]

            ax.scatter(
                points,
                [0] * len(points),
                label="Additional Points",
                zorder=3,
                c=colors,
                marker="D",
                edgecolor="k",
            )
        ax.set_title(
            f"Total={total_points}, Unique={unique_values}, Outliers={num_outliers}, StdDev={std_dev:.2f}"
        )
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle="--", linewidth=0.5)
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.show()


def plot_ip_boxplots(
    df: pd.DataFrame,
    category_column_name: str = None,
    num_cols: int = 2,
    figwidth: int = 10,
    box_width: float = 0.2,
    both_names=True,
    height_per_subplot: float = 2.0,
    add_points: pd.DataFrame = None,
    add_points_col: list = ["red"],
) -> None:
    """
    Generate infill-point boxplots (ip-boxplots). A separate ip-boxplot is generated for each numerical column in a DataFrame, arranged in a grid.
    Each subplot has its own scale, similar to how histograms are shown in plot_histograms().
    Additional points can be added and highlighted in red.

    Args:
        df (pd.DataFrame):
            DataFrame containing the data to plot.
        category_column_name (str, optional):
            Column name for categorical grouping. Defaults to None.
        num_cols (int, optional):
            Number of columns in the subplot grid. Defaults to 2.
        figwidth (int, optional):
            Width of the entire figure. Defaults to 10.
        box_width (float, optional):
            Width of the boxplots. Defaults to 0.2.
        both_names (bool, optional):
            Whether to show both variable names and categories in titles. Defaults to True.
        height_per_subplot (float, optional):
            Height per subplot row. Defaults to 2.0.
        add_points (pd.DataFrame, optional):
            DataFrame containing additional points to highlight. Defaults to None.
        add_points_col (list, optional):
            List of colors for the additional points. Defaults to ["red"].

    Returns:
        None

    Examples:
        >>> import pandas as pd
        >>> from spotoptim.eda.plots import plot_ip_boxplots
        >>> data = {'A': [1, 2, 2, 3, 4, 5, 100], 'B': [10, 10, 10, 10, 10, 10, 10]}
        >>> df = pd.DataFrame(data)
        >>> plot_ip_boxplots(df, num_cols=1)
        >>> # Example with multiple added points and colors
        >>> add_points = pd.DataFrame({'A': [1.5, 3.5], 'B': [10, 10]})
        >>> plot_ip_boxplots(df, add_points=add_points, add_points_col=["red", "blue"])
    """
    if df.ndim == 1:
        df = df.to_frame()
    numerical_columns = df.select_dtypes(include="number").columns.tolist()
    num_plots = len(numerical_columns)
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(figwidth, num_rows * height_per_subplot),
    )
    # Ensure axes is always an array
    if num_rows == 1 and num_cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    for i, col in enumerate(numerical_columns):
        ax = axes[i]
        if category_column_name and category_column_name in df.columns:
            unique_categories = sorted(df[category_column_name].dropna().unique())
            plot_data = [
                df.loc[df[category_column_name] == cat_value, col].dropna()
                for cat_value in unique_categories
            ]
        else:
            plot_data = [df[col].dropna()]
        ax.boxplot(
            plot_data,
            orientation="horizontal",
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", color="black"),
            medianprops=dict(color="red"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            flierprops=dict(marker="o", color="black", alpha=0.5),
            widths=box_width,
        )
        if add_points is not None and col in add_points.columns:
            if len(add_points) != len(add_points_col):
                raise ValueError(
                    f"Length of add_points ({len(add_points)}) and add_points_col ({len(add_points_col)}) must be the same."
                )

            points_data = add_points[[col]].copy()
            points_data["color"] = add_points_col
            points_data = points_data.dropna(subset=[col])

            points = points_data[col]
            colors = points_data["color"]

            ax.scatter(
                points,
                [1] * len(points),
                c=colors,
                marker="D",
                edgecolor="k",
                label="Additional Points",
                zorder=3,
            )
        if both_names:
            ax.set_title(col)
        else:
            ax.set_title(col)
        ax.set_xlabel("Value")
        if category_column_name and category_column_name in df.columns:
            ax.set_yticklabels(unique_categories)
            ax.set_ylabel(category_column_name)
        else:
            ax.set_yticklabels([""])
        ax.xaxis.grid(True, linestyle="--", linewidth=0.5)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5)
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()
