# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from spotoptim.mo.pareto import mo_xy_surface
import matplotlib.pyplot as plt


def test_mo_xy_surface_execution():
    # Setup dummy data
    np.random.seed(42)
    X = np.random.rand(50, 3)  # 3 features
    y1 = X[:, 0] + X[:, 1]
    y2 = X[:, 0] * X[:, 2]
    y = np.column_stack([y1, y2])

    models = []
    for i in range(y.shape[1]):
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y[:, i])
        models.append(model)

    bounds = [(0, 1), (0, 1), (0, 1)]
    target_names = ["Sum", "Prod"]
    feature_names = ["x1", "x2", "x3"]

    # Run the function (suppress plot show)
    try:
        # Mock plt.show to avoid blocking
        plt.show = lambda: None
        mo_xy_surface(
            models=models,
            bounds=bounds,
            target_names=target_names,
            feature_names=feature_names,
            resolution=10,  # Low resolution for speed
        )
    except Exception as e:
        pytest.fail(f"mo_xy_surface raised an exception: {e}")


if __name__ == "__main__":
    test_mo_xy_surface_execution()
