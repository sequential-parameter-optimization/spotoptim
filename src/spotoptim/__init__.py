"""SpotOptim - Sequential Parameter Optimization."""

from .SpotOptim import SpotOptim
from .surrogate import Kriging
from .data import DiabetesDataset, get_diabetes_dataloaders
from .tricands import tricands

__version__ = "0.0.3"
__all__ = [
    "SpotOptim",
    "Kriging",
    "DiabetesDataset",
    "get_diabetes_dataloaders",
    "tricands",
]


def hello() -> str:
    return "Hello from spotoptim!"
