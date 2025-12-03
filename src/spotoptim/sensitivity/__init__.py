"""
Sensitivity analysis module for spotoptim.

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

__all__ = [
    "plot_feature_importances",
    "generate_mdi",
    "generate_imp",
    "plot_importances",
    "plot_feature_scatter_matrix",
]
