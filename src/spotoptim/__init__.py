# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""SpotOptim - Sequential Parameter Optimization."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("spotoptim")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from .SpotOptim import SpotOptim, SpotOptimConfig, SpotOptimState
from .surrogate import Kriging, SimpleKriging, MLPSurrogate
from .nn import MLP, LinearRegressor
from .data import DiabetesDataset, get_diabetes_dataloaders
from .tricands import tricands
from .utils import (
    get_pca,
    plot_pca_scree,
    plot_pca1vs2,
    get_pca_topk,
    get_loading_scores,
    plot_loading_scores,
    TorchStandardScaler,
)

__all__ = [
    "SpotOptim",
    "SpotOptimConfig",
    "SpotOptimState",
    "Kriging",
    "SimpleKriging",
    "MLPSurrogate",
    "MLP",
    "LinearRegressor",
    "DiabetesDataset",
    "get_diabetes_dataloaders",
    "tricands",
    "get_pca",
    "plot_pca_scree",
    "plot_pca1vs2",
    "get_pca_topk",
    "get_loading_scores",
    "plot_loading_scores",
    "TorchStandardScaler",
]
