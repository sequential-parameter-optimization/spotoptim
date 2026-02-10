# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from spotoptim.hyperparameters.parameters import ParameterSet


def test_parameter_set_init():
    ps = ParameterSet()
    assert ps.var_name == []
    assert ps.var_type == []
    assert ps.bounds == []
    assert ps.var_trans == []
    assert ps.sample_default() == {}


def test_add_float():
    ps = ParameterSet()
    ps.add_float("lr", 0.001, 0.1, default=0.01, transform="log")

    assert ps.var_name == ["lr"]
    assert ps.var_type == ["float"]
    assert ps.bounds == [(0.001, 0.1)]
    assert ps.var_trans == ["log"]
    assert ps.sample_default() == {"lr": 0.01}


def test_add_int():
    ps = ParameterSet()
    ps.add_int("depth", 1, 10, default=5)

    assert ps.var_name == ["depth"]
    assert ps.var_type == ["int"]
    assert ps.bounds == [(1, 10)]
    assert ps.var_trans == [None]
    assert ps.sample_default() == {"depth": 5}


def test_add_factor():
    ps = ParameterSet()
    choices = ["adam", "sgd"]
    ps.add_factor("optimizer", choices, default="adam")

    assert ps.names() == ["optimizer"]
    assert ps.var_type == ["factor"]
    assert ps.bounds == [choices]
    assert ps.sample_default() == {"optimizer": "adam"}


def test_chaining():
    ps = ParameterSet()
    ps.add_float("x", 0.0, 1.0).add_int("y", 1, 10).add_factor("z", ["a", "b"])

    assert ps.names() == ["x", "y", "z"]
    assert ps.var_type == ["float", "int", "factor"]
    assert len(ps.bounds) == 3
    assert ps.bounds[0] == (0.0, 1.0)
    assert ps.bounds[1] == (1, 10)
    assert ps.bounds[2] == ["a", "b"]


def test_mixed_parameters_and_defaults():
    ps = ParameterSet()
    ps.add_float("p1", 0.0, 1.0, default=0.5)
    ps.add_int("p2", 1, 10)  # No default
    ps.add_factor("p3", ["c1", "c2"], default="c1")

    defaults = ps.sample_default()
    assert len(defaults) == 2
    assert defaults["p1"] == 0.5
    assert "p2" not in defaults
    assert defaults["p3"] == "c1"


def test_repr():
    ps = ParameterSet()
    ps.add_float("x", 0.0, 1.0)
    repr_str = repr(ps)
    assert "ParameterSet(" in repr_str
    assert "x=" in repr_str
    assert "type='float'" in repr_str


def test_names_method():
    ps = ParameterSet()
    ps.add_float("a", 0, 1)
    ps.add_int("b", 0, 1)
    assert ps.names() == ["a", "b"]
