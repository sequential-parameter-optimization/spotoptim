# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Neural network models for spotoptim."""

__all__ = ["LinearRegressor", "MLP"]

_lazy_map = {
    "LinearRegressor": ("spotoptim.nn.linear_regressor", "LinearRegressor"),
    "MLP": ("spotoptim.nn.mlp", "MLP"),
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
