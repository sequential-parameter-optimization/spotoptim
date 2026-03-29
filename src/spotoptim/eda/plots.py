# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spotoptim.utils.stats import calculate_outliers


def _align_add_points(
    df: pd.DataFrame,
    add_points: pd.DataFrame,
) -> pd.DataFrame:
    """Align the column names of add_points to those of df.

    If add_points has the same number of numerical columns as df but different
    names, the column names of add_points are replaced with those of df so that
    both DataFrames can be plotted together without a key mismatch.

    Non-numerical columns in add_points are left unchanged.

    Args:
        df: Reference DataFrame whose numerical column names are authoritative.
        add_points: DataFrame of additional points to overlay on the plots.

    Returns:
        A copy of add_points with its numerical column names replaced by the
        corresponding numerical column names of df when the counts match, or
        the original add_points unchanged when the counts differ.

    Raises:
        TypeError: If df or add_points is not a pandas DataFrame.

    Examples:
        >>> import pandas as pd
        >>> from spotoptim.eda.plots import _align_add_points
        >>> df = pd.DataFrame({'x1': [1, 2], 'x2': [3, 4]})
        >>> ap = pd.DataFrame({'a': [1.5], 'b': [3.5]})
        >>> _align_add_points(df, ap).columns.tolist()
        ['x1', 'x2']
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not isinstance(add_points, pd.DataFrame):
        raise TypeError("add_points must be a pandas DataFrame.")

    df_num_cols = df.select_dtypes(include="number").columns.tolist()
    ap_num_cols = add_points.select_dtypes(include="number").columns.tolist()

    if len(df_num_cols) != len(ap_num_cols):
        # Cannot align; caller must handle column-by-column matching
        return add_points.copy()

    rename_map = dict(zip(ap_num_cols, df_num_cols))
    return add_points.rename(columns=rename_map)


def plot_ip_histograms(
    df: pd.DataFrame,
    bins: int = 10,
    num_cols: int = 2,
    figwidth: int = 10,
    thrs_unique: int = 5,
    add_points: pd.DataFrame = None,
    add_points_col: list = ["red"],
) -> None:
    """Generate infill-point histograms for each numerical column in a DataFrame.

    A separate histogram is created for each numerical column of df and arranged
    in a grid. The title of each subplot shows the total point count, the number
    of unique values, the number of outliers detected via the IQR method, and
    the standard deviation. Columns with fewer unique values than thrs_unique are
    coloured differently to draw attention to low-variability features.

    If add_points is provided and its numerical columns differ from those of df
    in name but agree in count, the column names of add_points are silently
    replaced by those of df before plotting. This ensures that infill points
    drawn from a differently named search space are always overlaid correctly.

    Args:
        df: DataFrame containing the data to plot.
        bins: Number of bins for the histograms. Defaults to 10.
        num_cols: Number of columns in the subplot grid. Defaults to 2.
        figwidth: Width of the entire figure in inches. Defaults to 10.
        thrs_unique: Threshold for unique values below which the histogram bar
            colour switches from lightblue to lightcoral. Defaults to 5.
        add_points: DataFrame containing additional points to highlight with
            diamond markers at y = 0. Defaults to None.
        add_points_col: List of colours, one per row of add_points. Defaults
            to ["red"].

    Returns:
        None

    Raises:
        ValueError: If the number of rows in add_points does not equal the
            length of add_points_col.

    Examples:
        ```{python}
        import pandas as pd
        from spotoptim.eda.plots import plot_ip_histograms
        data = {'A': [1, 2, 2, 3, 4, 5, 100], 'B': [10, 10, 10, 10, 10, 10, 10]}
        df = pd.DataFrame(data)
        plot_ip_histograms(df, bins=5, num_cols=1, thrs_unique=3)
        # Example with multiple added points and colors
        add_points = pd.DataFrame({'A': [1.5, 3.5], 'B': [10, 10]})
        plot_ip_histograms(df, add_points=add_points, add_points_col=["red", "blue"])
        # Example: add_points with different column names are aligned automatically
        add_points_renamed = pd.DataFrame({'x': [1.5, 3.5], 'y': [10, 10]})
        plot_ip_histograms(df, add_points=add_points_renamed, add_points_col=["green", "orange"])
        ```

    References:
        Bartz-Beielstein, T. (2025). Multi-Objective Optimization and Hyperparameter
        Tuning With Desirability Functions. arXiv preprint arXiv:2503.23595.
        https://arxiv.org/abs/2503.23595
    """
    if add_points is not None:
        add_points = _align_add_points(df, add_points)

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
                    f"Length of add_points ({len(add_points)}) and "
                    f"add_points_col ({len(add_points_col)}) must be the same."
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
            f"Total={total_points}, Unique={unique_values}, "
            f"Outliers={num_outliers}, StdDev={std_dev:.2f}"
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
    both_names: bool = True,
    height_per_subplot: float = 2.0,
    add_points: pd.DataFrame = None,
    add_points_col: list = ["red"],
) -> None:
    """Generate infill-point boxplots for each numerical column in a DataFrame.

    A separate horizontal boxplot is created for each numerical column of df
    and arranged in a grid. Each subplot uses its own axis scale, mirroring the
    behaviour of plot_ip_histograms. An optional categorical grouping column
    splits the data into one box per category level.

    If add_points is provided and its numerical columns differ from those of df
    in name but agree in count, the column names of add_points are silently
    replaced by those of df before plotting. This ensures that infill points
    drawn from a differently named search space are always overlaid correctly.

    Args:
        df: DataFrame containing the data to plot.
        category_column_name: Column name for categorical grouping. When
            provided, one box is drawn per unique category. Defaults to None.
        num_cols: Number of columns in the subplot grid. Defaults to 2.
        figwidth: Width of the entire figure in inches. Defaults to 10.
        box_width: Width of each boxplot. Defaults to 0.2.
        both_names: When True, the subplot title shows both the variable name
            and, if applicable, the category column name. When False, only the
            variable name is shown. Defaults to True.
        height_per_subplot: Height in inches allocated to each subplot row.
            Defaults to 2.0.
        add_points: DataFrame containing additional points to highlight with
            diamond markers. Defaults to None.
        add_points_col: List of colours, one per row of add_points. Defaults
            to ["red"].

    Returns:
        None

    Raises:
        ValueError: If the number of rows in add_points does not equal the
            length of add_points_col.

    Examples:
        ```{python}
        import pandas as pd
        from spotoptim.eda.plots import plot_ip_boxplots
        data = {'A': [1, 2, 2, 3, 4, 5, 100], 'B': [10, 10, 10, 10, 10, 10, 10]}
        df = pd.DataFrame(data)
        plot_ip_boxplots(df, num_cols=1)
        # Example with multiple added points and colors
        add_points = pd.DataFrame({'A': [1.5, 3.5], 'B': [10, 10]})
        plot_ip_boxplots(df, add_points=add_points, add_points_col=["red", "blue"])
        # Example: add_points with different column names are aligned automatically
        add_points_renamed = pd.DataFrame({'x': [1.5, 3.5], 'y': [10, 10]})
        plot_ip_boxplots(df, add_points=add_points_renamed, add_points_col=["green", "orange"])
        ```

    References:
        Bartz-Beielstein, T. (2025). Multi-Objective Optimization and Hyperparameter
        Tuning With Desirability Functions. arXiv preprint arXiv:2503.23595.
        https://arxiv.org/abs/2503.23595
    """
    if df.ndim == 1:
        df = df.to_frame()

    if add_points is not None:
        add_points = _align_add_points(df, add_points)

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
            unique_categories = None
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
                    f"Length of add_points ({len(add_points)}) and "
                    f"add_points_col ({len(add_points_col)}) must be the same."
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
        if both_names and category_column_name and category_column_name in df.columns:
            ax.set_title(f"{col} by {category_column_name}")
        else:
            ax.set_title(col)
        ax.set_xlabel("Value")
        if unique_categories is not None:
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
