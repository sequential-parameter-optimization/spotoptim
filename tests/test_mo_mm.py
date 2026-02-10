# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from spotdesirability import DOverall, DMax
from spotoptim.function.mo import mo_conv2_max
from spotoptim.sampling.mm import mmphi_intensive
from spotoptim.mo.mo_mm import mo_mm_desirability_function


def test_mo_mm_desirability_function():
    # Setup based on docstring example
    np.random.seed(42)
    X_base = np.random.rand(50, 2)  # Reduced size for speed
    y = mo_conv2_max(X_base)

    models = []
    for i in range(y.shape[1]):
        model = RandomForestRegressor(
            n_estimators=10, random_state=42
        )  # Reduced estimators for speed
        model.fit(X_base, y[:, i])
        models.append(model)

    phi_base, J_base, d_base = mmphi_intensive(X_base, q=2, p=2)

    d_funcs = []
    for i in range(y.shape[1]):
        d_func = DMax(low=np.min(y[:, i]), high=np.max(y[:, i]))
        d_funcs.append(d_func)

    D_overall = DOverall(*d_funcs)
    x_test = np.random.rand(2)

    # Test with mm_objective=False
    neg_D, objectives = mo_mm_desirability_function(
        x_test, models, X_base, J_base, d_base, phi_base, D_overall, mm_objective=False
    )

    assert isinstance(neg_D, float)
    assert neg_D <= 0  # Desirability is positive, so negative desirability is negative
    assert len(objectives) == 2

    # Test with mm_objective=True
    # Note: D_overall needs to handle the extra objective if we want it to be part of the calculation
    # But the function calculates it and appends it to predictions.
    # If D_overall expects 2 inputs but gets 3, it might fail or ignore the 3rd depending on implementation.
    # The docstring says: "The combined desirability... is then computed using the provided DOverall object."
    # If mm_objective is True, predictions will have 3 elements.
    # Let's add a dummy desirability for the MM objective to D_overall for this part of the test.

    d_funcs_mm = d_funcs + [DMax(low=0, high=1)]  # Dummy DMax for MM
    D_overall_mm = DOverall(*d_funcs_mm)

    neg_D_mm, objectives_mm = mo_mm_desirability_function(
        x_test,
        models,
        X_base,
        J_base,
        d_base,
        phi_base,
        D_overall_mm,
        mm_objective=True,
    )

    assert isinstance(neg_D_mm, float)
    assert neg_D_mm <= 0
    assert len(objectives_mm) == 3
