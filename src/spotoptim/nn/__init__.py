# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Neural network models for spotoptim."""

__all__ = [
    "LinearRegressor",
    "MLP",
    "ManyToManyRNN",
    "ManyToManyRNNRegressor",
    "get_activation",
    "optimizer_handler",
]

_lazy_map = {
    "LinearRegressor": ("spotoptim.nn.linear_regressor", "LinearRegressor"),
    "MLP": ("spotoptim.nn.mlp", "MLP"),
    "ManyToManyRNN": ("spotoptim.nn.many_to_many_rnn", "ManyToManyRNN"),
    "ManyToManyRNNRegressor": (
        "spotoptim.nn.many_to_many_rnn",
        "ManyToManyRNNRegressor",
    ),
    "get_activation": ("spotoptim.nn.many_to_many_rnn", "get_activation"),
    "optimizer_handler": ("spotoptim.nn.optimizer", "optimizer_handler"),
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
