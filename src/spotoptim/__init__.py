"""SpotOptim - Sequential Parameter Optimization."""

from .SpotOptim import SpotOptim
from .surrogate import Kriging

__version__ = "0.0.3"
__all__ = ["SpotOptim", "Kriging"]


def hello() -> str:
    return "Hello from spotoptim!"
