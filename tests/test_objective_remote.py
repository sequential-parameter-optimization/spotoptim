# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import requests
import warnings
from spotoptim.function.remote import objective_remote


def test_objective_remote():
    """Integration check: remote objective returns finite values with correct shape."""

    test_X = np.array(
        [
            [0.48, 0.4, 0.38, 0.5, 0.62, 0.344, 0.4, 0.37, 0.38, 0.2],
            [0.48, 0.4, 0.38, 0.5, 0.62, 0.344, 0.4, 0.37, 0.38, 0.2],
            [0.48, 0.4, 0.38, 0.5, 0.62, 0.344, 0.4, 0.37, 0.38, 0.2],
        ]
    )

    try:
        result = objective_remote(test_X)
    except (
        requests.exceptions.RequestException
    ) as exc:  # pragma: no cover - network dependent
        warnings.warn(f"Could not connect to remote server: {exc}")
        return

    # Basic shape/type checks
    assert isinstance(result, np.ndarray), "Result must be a numpy array"
    assert result.shape == (3,), f"Expected shape (3,), got {result.shape}"

    # All values should be finite and non-negative
    assert np.all(
        np.isfinite(result.astype(float))
    ), "All returned values must be finite"
    assert np.all(result >= 0), "Objective values should be non-negative"

    # For identical inputs we expect similar costs; allow some jitter if the
    # remote service adds noise. A wide tolerance keeps the test robust.
    assert (
        np.max(result) - np.min(result) < 10
    ), "Costs for identical rows should not vary wildly"
