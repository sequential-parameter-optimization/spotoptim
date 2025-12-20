import numpy as np
import pytest
from spotoptim import SpotOptim

def noisy_sphere(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1) + np.random.normal(0, 0.1, X.shape[0])

def test_ocba_negative_repeats_repro_seed_86():
    """
    Test that SpotOptim handles OCBA computation without generating negative repeats.
    This specifically tests seed 86 which was found to fail.
    """
    seed = 86
    np.random.seed(seed)
    opt = SpotOptim(
        fun=noisy_sphere,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=25,
        n_initial=10,
        repeats_initial=2,
        ocba_delta=5,
        seed=seed,
        verbose=False
    )
    
    # This should not raise ValueError: repeats may not contain negative values
    try:
        result = opt.optimize()
        assert result.success
    except ValueError as e:
        if "repeats may not contain negative values" in str(e):
            pytest.fail(f"OCBA failed with negative repeats on seed {seed}: {e}")
        raise e

def test_ocba_various_seeds():
    """
    Test OCBA with various seeds to catch other potential failures.
    """
    for seed in [12, 42, 86, 99]:
        opt = SpotOptim(
            fun=noisy_sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=15,
            n_initial=10,
            repeats_initial=2,
            ocba_delta=5,
            seed=seed,
            verbose=False
        )
        try:
            opt.optimize()
        except ValueError as e:
            if "repeats may not contain negative values" in str(e):
                pytest.fail(f"OCBA failed with negative repeats on seed {seed}")
            raise e
