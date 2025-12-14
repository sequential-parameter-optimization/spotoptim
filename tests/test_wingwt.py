import numpy as np
import pytest
from spotoptim.function.so import wingwt

def test_wingwt_basics():
    """Test basic functionality of wingwt function."""
    # Baseline Cessna C172 configuration (from Forrester et al. 2008)
    # [Sw, Wfw, A, L, q, l, Rtc, Nz, Wdg]
    x_base = np.array([0.48, 0.4, 0.38, 0.5, 0.62, 0.344, 0.4, 0.37, 0.38])
    
    # Expected value (approximate)
    # Calculated from the formula with these inputs:
    # Sw=174, Wfw=252, A=7.52, L=0, q=34, l=0.672, Rtc=0.12, Nz=3.8, Wdg=2004
    # Result roughly around 306 lbs
    
    result = wingwt(x_base)
    
    assert result.shape == (1,)
    assert isinstance(result, np.ndarray)
    assert 233.0 < result[0] < 234.0, f"Expected ~233.91, got {result[0]}"

def test_wingwt_batch():
    """Test batch evaluation."""
    n_samples = 5
    rng = np.random.RandomState(42)
    X = rng.rand(n_samples, 9)
    
    results = wingwt(X)
    
    assert results.shape == (n_samples,)
    assert np.all(results > 0) # Weight should be positive

def test_wingwt_validation():
    """Test input validation."""
    # Wrong dimensions (8 inputs)
    X_bad = np.zeros((1, 8))
    with pytest.raises(ValueError, match="wingwt expects 9 or 10 features"):
        wingwt(X_bad)

def test_wingwt_painit():
    """Test with 10 variables (painted)."""
    # Baseline with W_p (10th variable)
    x_base = np.array([0.48, 0.4, 0.38, 0.5, 0.62, 0.344, 0.4, 0.37, 0.38, 0.2]) # W_p = 0.2 scaled
    
    # Painted weight should be higher than unpainted
    w_painted = wingwt(x_base)[0]
    
    x_base_unpainted = x_base[:9]
    w_unpainted = wingwt(x_base_unpainted)[0]
    
    assert w_painted > w_unpainted
    
    # Verify the difference matches the formula: Sw * Wp
    # Sw = 0.48 * 50 + 150 = 174
    # Wp = 0.2 * 0.02 + 0.06 = 0.064
    # Expected Diff = 174 * 0.064 = 11.136
    
    diff = w_painted - w_unpainted
    assert np.isclose(diff, 174 * (0.2 * (0.08 - 0.06) + 0.06), rtol=1e-5)

def test_wingwt_bounds():
    """Test boundary conditions."""
    # All zeros
    x_min = np.zeros(9)
    w_min = wingwt(x_min)[0]
    assert w_min > 0
    
    # All ones
    x_max = np.ones(9)
    w_max = wingwt(x_max)[0]
    assert w_max > 0
