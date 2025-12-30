"""SpotOptim - Sequential Parameter Optimization."""

from .SpotOptim import SpotOptim
from .surrogate import Kriging
from .data import DiabetesDataset, get_diabetes_dataloaders
from .tricands import tricands
from .utils import (
    get_pca,
    plot_pca_scree,
    plot_pca1vs2,
    get_pca_topk,
    get_loading_scores,
    plot_loading_scores,
)

__version__ = "0.0.3"
__all__ = [
    "SpotOptim",
    "Kriging",
    "DiabetesDataset",
    "get_diabetes_dataloaders",
    "tricands",
    "get_pca",
    "plot_pca_scree",
    "plot_pca1vs2",
    "get_pca_topk",
    "get_loading_scores",
    "plot_loading_scores",
]


def hello() -> str:
    return "Hello from spotoptim!"
