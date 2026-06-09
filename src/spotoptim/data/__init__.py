# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Data utilities for spotoptim package.

This module provides ready-to-use datasets and data loaders for common
machine learning tasks.
"""

from .base import Config, FileConfig

__all__ = ["DiabetesDataset", "get_diabetes_dataloaders", "Config", "FileConfig"]

_lazy_map = {
    "DiabetesDataset": ("spotoptim.data.diabetes", "DiabetesDataset"),
    "get_diabetes_dataloaders": ("spotoptim.data.diabetes", "get_diabetes_dataloaders"),
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
