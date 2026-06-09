# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Contract test: the removed parallel parameters are hard-rejected.

After the parallel-evaluation subsystem was removed, ``SpotOptim`` must not
accept the ``n_jobs`` or ``eval_batch_size`` constructor arguments. Passing
either should raise ``TypeError`` (Python's unknown-keyword rejection). This
test pins that contract so a future ``**kwargs`` passthrough cannot silently
re-admit the dead parameters.
"""

import numpy as np
import pytest

from spotoptim import SpotOptim


def _objective(X):
    return np.sum(X**2, axis=1)


@pytest.mark.parametrize("param", ["n_jobs", "eval_batch_size"])
def test_removed_parallel_param_raises_type_error(param):
    """Constructing SpotOptim with a removed parallel param raises TypeError."""
    with pytest.raises(TypeError, match=param):
        SpotOptim(fun=_objective, bounds=[(-5.0, 5.0), (-5.0, 5.0)], **{param: 2})
