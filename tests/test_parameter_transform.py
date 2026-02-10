# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from spotoptim.hyperparameters.parameters import ParameterSet
from spotoptim.SpotOptim import SpotOptim
import numpy as np
import pytest

def test_parameter_set_transform():
    ps = ParameterSet()
    ps.add_float("x", 1.0, 10.0, transform="log")
    ps.add_float("y", 1.0, 10.0, transform="log(x)")
    ps.add_float("z", 1.0, 10.0, transform="pow(x, 2)")
    
    assert ps.var_trans == ["log", "log(x)", "pow(x, 2)"]
    
    # Check repr
    r = repr(ps)
    assert "transform='log'" in r
    assert "transform='log(x)'" in r
    assert "transform='pow(x, 2)'" in r

def test_spotoptim_dynamic_transforms():
    spot = SpotOptim(fun=lambda x: np.sum(x), bounds=[(1, 10)])
    
    # Test log(x)
    x = 10.0
    val = spot.transform_value(x, "log(x)")
    assert np.isclose(val, np.log(x))
    inv = spot.inverse_transform_value(val, "log(x)")
    assert np.isclose(inv, x)

    # Test sqrt(x)
    x = 4.0
    val = spot.transform_value(x, "sqrt(x)")
    assert np.isclose(val, 2.0)
    inv = spot.inverse_transform_value(val, "sqrt(x)")
    assert np.isclose(inv, x)

    # Test pow(x, 2)
    x = 3.0
    val = spot.transform_value(x, "pow(x, 2)")
    assert np.isclose(val, 9.0)
    inv = spot.inverse_transform_value(val, "pow(x, 2)")
    assert np.isclose(inv, x)

    # Test pow(x, 3)
    x = 2.0
    val = spot.transform_value(x, "pow(x, 3)")
    assert np.isclose(val, 8.0)
    inv = spot.inverse_transform_value(val, "pow(x, 3)")
    assert np.isclose(inv, x)

    # Test log(x, 10)
    x = 100.0
    val = spot.transform_value(x, "log(x, 10)")
    assert np.isclose(val, 2.0)
    inv = spot.inverse_transform_value(val, "log(x, 10)")
    assert np.isclose(inv, x)
    
def test_unknown_transform():
    spot = SpotOptim(fun=lambda x: x, bounds=[(1, 10)])
    with pytest.raises(ValueError):
        spot.transform_value(1.0, "unknown_func(x)")
