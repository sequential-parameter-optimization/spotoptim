"""Utility functions for spotoptim."""

from .boundaries import get_boundaries, map_to_original_scale
from .mapping import map_lr
from .stats import normalize_X, calculate_outliers, get_combinations
from .eval import mo_eval_models, mo_cv_models
from .pca import (
    get_pca,
    plot_pca_scree,
    plot_pca1vs2,
    get_pca_topk,
    get_loading_scores,
    plot_loading_scores,
)

__all__ = [
    "get_boundaries",
    "map_to_original_scale",
    "map_lr",
    "normalize_X",
    "calculate_outliers",
    "get_combinations",
    "mo_eval_models",
    "mo_cv_models",
    "get_pca",
    "plot_pca_scree",
    "plot_pca1vs2",
    "get_pca_topk",
    "get_loading_scores",
    "plot_loading_scores",
]
