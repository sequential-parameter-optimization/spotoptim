# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Utility functions for spotoptim."""

from .boundaries import get_boundaries, map_to_original_scale
from .mapping import map_lr
from .eval import mo_eval_models, mo_cv_models
from .file import get_experiment_filename, get_internal_datasets_folder

__all__ = [
    "get_boundaries",
    "map_to_original_scale",
    "map_lr",
    "normalize_X",
    "calculate_outliers",
    "get_combinations",
    "mo_eval_models",
    "mo_cv_models",
    "get_experiment_filename",
    "get_internal_datasets_folder",
    "get_pca",
    "plot_pca_scree",
    "plot_pca1vs2",
    "get_pca_topk",
    "get_loading_scores",
    "plot_loading_scores",
    "TorchStandardScaler",
    "seed_everything",
]

_lazy_map = {
    # stats module (matplotlib/seaborn/statsmodels imported lazily inside the
    # plotting/regression helpers; these three are pure numpy/pandas/scipy)
    "normalize_X": ("spotoptim.utils.stats", "normalize_X"),
    "calculate_outliers": ("spotoptim.utils.stats", "calculate_outliers"),
    "get_combinations": ("spotoptim.utils.stats", "get_combinations"),
    # pca module (matplotlib/seaborn imported lazily inside the plot helpers)
    "get_pca": ("spotoptim.utils.pca", "get_pca"),
    "plot_pca_scree": ("spotoptim.utils.pca", "plot_pca_scree"),
    "plot_pca1vs2": ("spotoptim.utils.pca", "plot_pca1vs2"),
    "get_pca_topk": ("spotoptim.utils.pca", "get_pca_topk"),
    "get_loading_scores": ("spotoptim.utils.pca", "get_loading_scores"),
    "plot_loading_scores": ("spotoptim.utils.pca", "plot_loading_scores"),
    # scaler (pulls torch)
    "TorchStandardScaler": ("spotoptim.utils.scaler", "TorchStandardScaler"),
    # seed (pulls torch)
    "seed_everything": ("spotoptim.utils.seed", "seed_everything"),
}


def __getattr__(name: str):
    if name in _lazy_map:
        module_path, attr = _lazy_map[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
