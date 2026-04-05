# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Analysis utilities for variable importance and sensitivity."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from spotoptim.core.protocol import SpotOptimProtocol


def get_importance(optimizer: SpotOptimProtocol) -> List[float]:
    """Calculate variable importance scores.

    Importance is computed as the normalized sensitivity of each parameter
    based on the variation in objective values across the evaluated points.

    Args:
        optimizer: SpotOptim instance.

    Returns:
        List[float]: Importance scores for each dimension (0-100 scale).
    """
    if optimizer.X_ is None or optimizer.y_ is None or len(optimizer.y_) < 3:
        # Not enough data to compute importance
        return [0.0] * len(optimizer.all_lower)

    # Use full-dimensional data
    X_full = optimizer.X_
    if optimizer.red_dim:
        X_full = np.array(
            [optimizer.to_all_dim(x.reshape(1, -1))[0] for x in optimizer.X_]
        )

    # Calculate sensitivity for each dimension
    sensitivities = []
    for i in range(X_full.shape[1]):
        x_i = X_full[:, i]

        # Handle factor variables: map strings to integers
        if hasattr(optimizer, "_factor_maps") and i in optimizer._factor_maps:
            # _factor_maps[i] is {int: str}, we need {str: int}
            str_to_int = {v: k for k, v in optimizer._factor_maps[i].items()}
            try:
                x_i = np.array([str_to_int.get(val, -1) for val in x_i])
            except Exception:
                sensitivities.append(0.0)
                continue
        else:
            # Ensure numeric type for non-factors
            try:
                x_i = x_i.astype(float)
            except ValueError:
                sensitivities.append(0.0)
                continue

        # Skip if no variation in this dimension
        if np.std(x_i) < 1e-10:
            sensitivities.append(0.0)
            continue

        # Compute correlation with objective
        try:
            correlation = np.abs(np.corrcoef(x_i, optimizer.y_)[0, 1])
            if np.isnan(correlation):
                correlation = 0.0
        except Exception:
            correlation = 0.0

        sensitivities.append(correlation)

    # Normalize to percentage
    total = sum(sensitivities)
    if total > 0:
        importance = [(s / total) * 100 for s in sensitivities]
    else:
        importance = [0.0] * len(sensitivities)

    return importance


def sensitivity_spearman(optimizer: SpotOptimProtocol) -> None:
    """Compute and print Spearman correlation between parameters and objective values.

    This method analyzes the sensitivity of the objective function to each
    hyperparameter by computing Spearman rank correlations.

    Significance levels:
        * ***: p < 0.001 (highly significant)
        * **: p < 0.01 (significant)
        * *: p < 0.05 (marginally significant)

    Args:
        optimizer: SpotOptim instance.
    """
    try:
        from scipy.stats import spearmanr
    except ImportError:
        raise ImportError(
            "scipy is required for sensitivity_spearman(). "
            "Install it with: pip install scipy"
        )

    if optimizer.X_ is None or optimizer.y_ is None:
        raise ValueError("No optimization data available. Run optimize() first.")

    # Get optimization history and parameters
    history = optimizer.y_
    all_params = optimizer.X_

    # Get parameter names
    param_names = (
        optimizer.var_name
        if optimizer.var_name
        else [f"x{i}" for i in range(optimizer.n_dim)]
    )

    print("\nSensitivity Analysis (Spearman Correlation):")
    print("-" * 50)

    for param_idx in range(optimizer.n_dim):
        name = param_names[param_idx]
        param_values = all_params[:, param_idx]

        # Check if it's a factor variable
        var_type = optimizer.var_type[param_idx] if optimizer.var_type else "float"

        if var_type == "factor":
            # For categorical variables, skip correlation
            print(f"  {name:20s}: (categorical variable, use visual inspection)")
            continue

        # Check if parameter has log transformation
        var_trans = optimizer.var_trans[param_idx] if optimizer.var_trans else None

        # Compute correlation based on transformation
        if var_trans in ["log10", "log", "ln"]:
            # For log-transformed parameters, use log-space correlation
            try:
                param_values_numeric = param_values.astype(float)
                # Filter out non-positive values
                valid_mask = (param_values_numeric > 0) & (history > 0)
                if valid_mask.sum() < 3:
                    print(
                        f"  {name:20s}: (insufficient valid data for log correlation)"
                    )
                    continue

                corr, p_value = spearmanr(
                    np.log10(param_values_numeric[valid_mask]),
                    np.log10(history[valid_mask]),
                )
            except (ValueError, TypeError):
                print(f"  {name:20s}: (error computing log correlation)")
                continue
        else:
            # For integer/float parameters, direct correlation
            try:
                param_values_numeric = param_values.astype(float)
                corr, p_value = spearmanr(param_values_numeric, history)
            except (ValueError, TypeError):
                print(f"  {name:20s}: (error computing correlation)")
                continue

        # Determine significance level
        if p_value < 0.001:
            significance = " ***"
        elif p_value < 0.01:
            significance = " **"
        elif p_value < 0.05:
            significance = " *"
        else:
            significance = ""

        print(f"  {name:20s}: {corr:+.3f} (p={p_value:.3f}){significance}")


def get_stars(input_list: list) -> list:
    """Converts a list of values to a list of stars.

    Thresholds: >99: ***, >75: **, >50: *, >10: .

    Args:
        input_list (list): A list of importance scores (0-100).

    Returns:
        list: A list of star strings.
    """
    output_list = []
    for value in input_list:
        if value > 99:
            output_list.append("***")
        elif value > 75:
            output_list.append("**")
        elif value > 50:
            output_list.append("*")
        elif value > 10:
            output_list.append(".")
        else:
            output_list.append("")
    return output_list
