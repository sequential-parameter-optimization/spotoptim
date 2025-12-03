"""
Inspection (sensitivity analysis) module for spotoptim.

Provides feature importance utilities based on impurity (MDI) and
permutation importance, plus plotting helpers.
"""

from .importance import (
    plot_feature_importances,
    generate_mdi,
    generate_imp,
    plot_importances,
    plot_feature_scatter_matrix,
)

from .predictions import plot_actual_vs_predicted

__all__ = [
    "plot_feature_importances",
    "generate_mdi",
    "generate_imp",
    "plot_importances",
    "plot_feature_scatter_matrix",
    "plot_actual_vs_predicted",
]
