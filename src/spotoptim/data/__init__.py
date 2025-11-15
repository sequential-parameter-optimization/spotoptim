"""Data utilities for spotoptim package.

This module provides ready-to-use datasets and data loaders for common
machine learning tasks.
"""

from .diabetes import DiabetesDataset, get_diabetes_dataloaders

__all__ = ["DiabetesDataset", "get_diabetes_dataloaders"]
