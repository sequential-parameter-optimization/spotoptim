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
from .surrogate import Kriging, SimpleKriging
from .tricands import tricands

__all__ = [
    "SpotOptim",
    "SpotOptimConfig",
    "SpotOptimState",
    "Kriging",
    "SimpleKriging",
    "tricands",
]
