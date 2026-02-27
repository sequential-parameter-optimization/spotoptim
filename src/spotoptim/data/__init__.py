# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Data utilities for spotoptim package.

This module provides ready-to-use datasets and data loaders for common
machine learning tasks.
"""

from .diabetes import DiabetesDataset, get_diabetes_dataloaders
from .base import Config, FileConfig

__all__ = ["DiabetesDataset", "get_diabetes_dataloaders", "Config", "FileConfig"]

