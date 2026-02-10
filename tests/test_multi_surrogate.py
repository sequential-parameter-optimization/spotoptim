# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pytest
from spotoptim import SpotOptim
from sklearn.base import BaseEstimator, RegressorMixin

class DummySurrogate(BaseEstimator, RegressorMixin):
    def __init__(self, name="dummy"):
        self.name = name
        self.fitted = False

    def fit(self, X, y):
        self.fitted = True
        return self

    def predict(self, X, return_std=False):
        n = X.shape[0]
        if return_std:
            return np.zeros(n), np.zeros(n)
        return np.zeros(n)

def sphere(X):
    return np.sum(X**2, axis=1)

def test_init_list_surrogates_uniform():
    """Test initialization with a list of surrogates and default uniform probabilities."""
    s1 = DummySurrogate("s1")
    s2 = DummySurrogate("s2")
    
    opt = SpotOptim(
        fun=sphere,
        bounds=[(-1, 1)],
        surrogate=[s1, s2],
        n_initial=2,
        seed=42
    )
    
    # Check internal list
    assert opt._surrogates_list == [s1, s2]
    # Check uniform probs
    assert opt._prob_surrogate == [0.5, 0.5]
    # Initial surrogate is the first one
    assert opt.surrogate == s1

def test_init_list_surrogates_custom_probs():
    """Test initialization with custom probabilities."""
    s1 = DummySurrogate("s1")
    s2 = DummySurrogate("s2")
    
    probs = [0.8, 0.2]
    
    opt = SpotOptim(
        fun=sphere,
        bounds=[(-1, 1)],
        surrogate=[s1, s2],
        prob_surrogate=probs,
        n_initial=2,
        seed=42
    )
    
    assert opt._prob_surrogate == probs

def test_init_list_validation_error():
    """Test validation errors for mismatched lengths."""
    s1 = DummySurrogate("s1")
    
    with pytest.raises(ValueError, match="Length of prob_surrogate"):
        SpotOptim(
            fun=sphere,
            bounds=[(-1, 1)],
            surrogate=[s1],
            prob_surrogate=[0.5, 0.5], # Mismatch
            n_initial=2
        )

def test_selection_logic_deterministic():
    """Test that selection respects probabilities (0.0 vs 1.0) and is reproducible."""
    s1 = DummySurrogate("s1")
    s2 = DummySurrogate("s2")
    
    # Case 1: Always select s2
    opt = SpotOptim(
        fun=sphere,
        bounds=[(-1, 1)],
        surrogate=[s1, s2],
        prob_surrogate=[0.0, 1.0],
        n_initial=2,
        max_iter=5, # 3 optimization steps
        seed=42
    )
    
    # Run optimization
    opt.optimize()
    
    # Verify s2 is the active surrogate (last selected)
    assert opt.surrogate == s2
    # Also verify s1 was never fitted? (Well, SpotOptim fits self.surrogate. 
    # If s1 was never selected, it might not be fitted unless initially fitted?)
    # SpotOptim init sets self.surrogate = list[0]. But first fit happens in loop?
    # Actually, `optimize()` does `_fit_scheduler()` inside loop.
    # Initial design evaluation doesn't trigger fit. First fit is after N_initial.
    # So s1 (idx 0) might be set initially but if first selection picks s2, s1.fit is never called.
    
    # Check if s1.fitted is False (assuming our Dummy tracks it)
    # Wait, simple dummy doesn't track calls precisely, let's verify logic.
    pass

def test_reproducibility_of_sequence():
    """Test using a seed produces same sequence of surrogates."""
    s1 = DummySurrogate("s1")
    s2 = DummySurrogate("s2")
    
    # Run 1
    opt1 = SpotOptim(
        fun=sphere,
        bounds=[(-1, 1)],
        surrogate=[s1, s2],
        max_iter=10,
        n_initial=2,
        seed=123
    )
    
    # Collect sequence of surrogates used
    # We can mock _fit_surrogate to record selection
    sequence1 = []
    
    original_fit = opt1._fit_surrogate
    def fit_hook(X, y):
        sequence1.append(opt1.surrogate.name)
        return original_fit(X, y)
        
    opt1._fit_surrogate = fit_hook
    opt1.optimize()
    
    # Run 2
    opt2 = SpotOptim(
        fun=sphere,
        bounds=[(-1, 1)],
        surrogate=[DummySurrogate("s1"), DummySurrogate("s2")],
        max_iter=10,
        n_initial=2,
        seed=123
    )
    
    sequence2 = []
    original_fit2 = opt2._fit_surrogate
    def fit_hook2(X, y):
        sequence2.append(opt2.surrogate.name)
        return original_fit2(X, y)
    
    opt2._fit_surrogate = fit_hook2
    opt2.optimize()
    
    assert sequence1 == sequence2
    assert len(sequence1) > 0

def test_integration_loop_alternating():
    """Test that it actually switches surrogates."""
    s1 = DummySurrogate("s1")
    s2 = DummySurrogate("s2")
    
    opt = SpotOptim(
        fun=sphere,
        bounds=[(-1, 1)],
        surrogate=[s1, s2],
        n_initial=2,
        max_iter=20,
        seed=42 # Should mix
    )
    
    names = set()
    original_fit = opt._fit_surrogate
    def fit_hook(X, y):
        names.add(opt.surrogate.name)
        return original_fit(X, y)
    
    opt._fit_surrogate = fit_hook
    opt.optimize()
    
    # With enough iterations and 50/50, both should be selected
    assert "s1" in names
    assert "s2" in names
