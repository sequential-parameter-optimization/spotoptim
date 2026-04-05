# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reporting utilities for displaying optimization results and search space design."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional

import numpy as np
from scipy.optimize import OptimizeResult
from tabulate import tabulate

if TYPE_CHECKING:
    from spotoptim.core.protocol import SpotOptimProtocol


def print_best(
    optimizer: SpotOptimProtocol,
    result: Optional[OptimizeResult] = None,
    transformations: Optional[List[Optional[Callable]]] = None,
    show_name: bool = True,
    precision: int = 4,
) -> None:
    """Print the best solution found during optimization.

    Args:
        optimizer: SpotOptim instance.
        result (OptimizeResult, optional): Optimization result object from optimize().
            If None, uses the stored best values from the optimizer. Defaults to None.
        transformations (list of callable, optional): List of transformation functions
            to apply to each parameter. Length must match number of dimensions.
            Defaults to None.
        show_name (bool, optional): Whether to display variable names. Defaults to True.
        precision (int, optional): Number of decimal places. Defaults to 4.
    """
    # Get values from result or stored attributes
    if result is not None:
        best_x = result.x
        best_y = result.fun
        n_evals = result.nfev
    else:
        if optimizer.best_x_ is None or optimizer.best_y_ is None:
            print("No optimization results available. Run optimize() first.")
            return
        best_x = optimizer.best_x_
        best_y = optimizer.best_y_
        n_evals = optimizer.counter

    # Expand to full dimensions if dimension reduction was applied
    if optimizer.red_dim:
        best_x_full = optimizer.to_all_dim(best_x.reshape(1, -1))[0]
    else:
        best_x_full = best_x

    # Map factor variables back to original string values
    best_x_full = optimizer.map_to_factor_values(best_x_full.reshape(1, -1))[0]

    # Determine variable names to use
    if show_name and optimizer.all_var_name is not None:
        var_names = optimizer.all_var_name
    else:
        var_names = [f"x{i}" for i in range(len(best_x_full))]

    # Validate transformations length
    if transformations is not None:
        if len(transformations) != len(best_x_full):
            raise ValueError(
                f"Length of transformations ({len(transformations)}) must match "
                f"number of dimensions ({len(best_x_full)})"
            )
    else:
        transformations = [None] * len(best_x_full)

    # Print header
    print("\nBest Solution Found:")
    print("-" * 50)

    # Print each parameter
    for i, (name, value, transform) in enumerate(
        zip(var_names, best_x_full, transformations)
    ):
        # Apply transformation if provided
        if transform is not None:
            try:
                display_value = transform(value)
            except Exception as e:
                print(f"Warning: Transformation failed for {name}: {e}")
                display_value = value
        else:
            display_value = value

        # Format based on variable type
        var_type = (
            optimizer.all_var_type[i] if i < len(optimizer.all_var_type) else "float"
        )

        if var_type == "int" or isinstance(display_value, (int, np.integer)):
            print(f"  {name}: {int(display_value)}")
        elif var_type == "factor" or isinstance(display_value, str):
            print(f"  {name}: {display_value}")
        else:
            print(f"  {name}: {display_value:.{precision}f}")

    # Print objective value and evaluations
    print(f"  Objective Value: {best_y:.{precision}f}")
    print(f"  Total Evaluations: {n_evals}")


def get_results_table(
    optimizer: SpotOptimProtocol,
    tablefmt: str = "github",
    precision: int = 4,
    show_importance: bool = False,
) -> str:
    """Get a comprehensive table string of optimization results.

    Args:
        optimizer: SpotOptim instance.
        tablefmt (str, optional): Table format for tabulate library. Defaults to 'github'.
        precision (int, optional): Number of decimal places. Defaults to 4.
        show_importance (bool, optional): Whether to include importance scores.
            Defaults to False.

    Returns:
        str: Formatted table string.
    """
    if optimizer.best_x_ is None or optimizer.best_y_ is None:
        return "No optimization results available. Run optimize() first."

    # Get best solution in full dimensions
    if optimizer.red_dim:
        best_x_full = optimizer.to_all_dim(optimizer.best_x_.reshape(1, -1))[0]
    else:
        best_x_full = optimizer.best_x_

    # Map factor variables back to original string values
    best_x_display = optimizer.map_to_factor_values(best_x_full.reshape(1, -1))[0]

    # Prepare all variable transformations (use all_var_trans if dimension reduction occurred)
    if optimizer.red_dim and hasattr(optimizer, "all_var_trans"):
        all_var_trans = optimizer.all_var_trans
    else:
        all_var_trans = optimizer.var_trans

    # Prepare table data
    table_data = {
        "name": (
            optimizer.all_var_name
            if optimizer.all_var_name
            else [f"x{i}" for i in range(len(best_x_display))]
        ),
        "type": (
            optimizer.all_var_type
            if optimizer.all_var_type
            else ["float"] * len(best_x_display)
        ),
        "default": [],
        "lower": [],
        "upper": [],
        "tuned": [],
        "transform": [t if t is not None else "-" for t in all_var_trans],
    }

    # Helper to format values
    def fmt_val(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.{precision}f}"
        return v

    # Process bounds, defaults, and tuned values
    for i in range(len(best_x_display)):
        var_type = table_data["type"][i]

        # Handle bounds and defaults based on variable type
        if var_type == "factor":
            # For factors, show original string values
            if i in optimizer._factor_maps:
                factor_map = optimizer._factor_maps[i]
                # Default is middle level logic (matching get_design_table)
                mid_idx = len(factor_map) // 2
                default_str = factor_map[mid_idx]

                table_data["lower"].append("-")
                table_data["upper"].append("-")
                table_data["default"].append(default_str)
            else:
                table_data["lower"].append("-")
                table_data["upper"].append("-")
                table_data["default"].append("N/A")
        else:
            table_data["lower"].append(fmt_val(optimizer._original_lower[i]))
            table_data["upper"].append(fmt_val(optimizer._original_upper[i]))
            # Default is midpoint logic
            default_val = (
                optimizer._original_lower[i] + optimizer._original_upper[i]
            ) / 2
            if var_type == "int":
                table_data["default"].append(int(default_val))
            else:
                table_data["default"].append(fmt_val(default_val))

        # Format tuned value
        tuned_val = best_x_display[i]
        if var_type == "int":
            table_data["tuned"].append(int(tuned_val))
        elif var_type == "factor":
            table_data["tuned"].append(str(tuned_val))
        else:
            table_data["tuned"].append(fmt_val(tuned_val))

    # Add importance if requested
    if show_importance:
        importance = optimizer.get_importance()
        table_data["importance"] = [f"{x:.2f}" for x in importance]
        table_data["stars"] = optimizer.get_stars(importance)

    # Generate table
    table = tabulate(
        table_data,
        headers="keys",
        tablefmt=tablefmt,
        numalign="right",
        stralign="right",
    )

    # Add interpretation if importance is shown
    if show_importance:
        table += "\n\nInterpretation: ***: >99%, **: >75%, *: >50%, .: >10%"

    return table


def get_design_table(
    optimizer: SpotOptimProtocol,
    tablefmt: str = "github",
    precision: int = 4,
) -> str:
    """Get a table string showing the search space design before optimization.

    Args:
        optimizer: SpotOptim instance.
        tablefmt (str, optional): Table format for tabulate library.
            Defaults to 'github'.
        precision (int, optional): Number of decimal places. Defaults to 4.

    Returns:
        str: Formatted table string.
    """
    # Prepare all variable transformations (use all_var_trans if dimension reduction occurred)
    if optimizer.red_dim and hasattr(optimizer, "all_var_trans"):
        all_var_trans = optimizer.all_var_trans
    else:
        all_var_trans = optimizer.var_trans

    # Prepare table data
    table_data = {
        "name": (
            optimizer.all_var_name
            if optimizer.all_var_name
            else [f"x{i}" for i in range(len(optimizer.all_lower))]
        ),
        "type": (
            optimizer.all_var_type
            if optimizer.all_var_type
            else ["float"] * len(optimizer.all_lower)
        ),
        "lower": [],
        "upper": [],
        "default": [],
        "transform": [t if t is not None else "-" for t in all_var_trans],
    }

    # Helper to format values
    def fmt_val(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.{precision}f}"
        return v

    # Process bounds and compute defaults (use original bounds for display)
    for i in range(len(optimizer._original_lower)):
        var_type = table_data["type"][i]

        if var_type == "factor":
            # For factors, show original string values
            if i in optimizer._factor_maps:
                factor_map = optimizer._factor_maps[i]
                # Default is middle level
                mid_idx = len(factor_map) // 2
                default_str = factor_map[mid_idx]
                table_data["lower"].append("-")
                table_data["upper"].append("-")
                table_data["default"].append(default_str)
            else:
                table_data["lower"].append("-")
                table_data["upper"].append("-")
                table_data["default"].append("N/A")
        else:
            table_data["lower"].append(fmt_val(optimizer._original_lower[i]))
            table_data["upper"].append(fmt_val(optimizer._original_upper[i]))
            # Default is midpoint
            default_val = (
                optimizer._original_lower[i] + optimizer._original_upper[i]
            ) / 2
            if var_type == "int":
                table_data["default"].append(int(default_val))
            else:
                table_data["default"].append(fmt_val(default_val))

    # Generate table
    table = tabulate(
        table_data,
        headers="keys",
        tablefmt=tablefmt,
        numalign="right",
        stralign="right",
    )

    return table
