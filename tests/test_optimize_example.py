import numpy as np
import pytest

from spotoptim import SpotOptim


def test_optimize_example_deterministic():
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1),
        bounds=[(-5, 5), (-5, 5)],
        n_initial=5,
        max_iter=20,
        seed=0,
        x0=np.array([0.0, 0.0]),
        verbose=False,
    )

    result = opt.optimize()

    assert result.success is True
    assert "maximum evaluations (20) reached" in result.message
    assert np.allclose(result.x, np.array([0.0, 0.0]), atol=1e-12)
    assert result.fun == pytest.approx(0.0, abs=1e-12)
