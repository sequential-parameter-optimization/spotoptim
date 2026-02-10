# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
from spotoptim import SpotOptim


def test_apply_penalty_na_example():
    """Test _apply_penalty_NA method with NaN and inf values."""
    opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1), bounds=[(-5, 5)])

    y_hist = np.array([1.0, 2.0, 3.0, 5.0])
    y_new = np.array([4.0, np.nan, np.inf])

    y_clean = opt._apply_penalty_NA(y_new, y_history=y_hist)

    # All values should be finite after penalty application
    assert np.all(np.isfinite(y_clean))

    # NaN/inf replaced with worst value from history + 3*std + noise
    # Should be larger than max finite value in history
    assert y_clean[1] > 5.0
    assert y_clean[2] > 5.0

    # First value should remain unchanged (it was already finite)
    assert y_clean[0] == 4.0
