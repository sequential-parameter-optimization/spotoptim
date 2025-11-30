import numpy as np
import pytest
from spotoptim.sampling.mm import mmphi_intensive, mmphi_intensive_update

def test_mmphi_intensive_basic():
    """Test with a simple 3-point plan."""
    X = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0]
    ])
    # q=2, p=2 (Euclidean)
    # Distances:
    # (0,0)-(0.5,0.5): sqrt(0.5^2 + 0.5^2) = sqrt(0.5) = 0.7071
    # (0.5,0.5)-(1,1): sqrt(0.5^2 + 0.5^2) = sqrt(0.5) = 0.7071
    # (0,0)-(1,1): sqrt(1^2 + 1^2) = sqrt(2) = 1.4142
    # Unique distances: [0.7071, 1.4142]
    # Multiplicities: [2, 1]
    # M = 3*2/2 = 3
    # Sum term = 2 * (0.7071)^(-2) + 1 * (1.4142)^(-2)
    #          = 2 * (1/0.5) + 1 * (1/2)
    #          = 2 * 2 + 0.5 = 4.5
    # intensive_phiq = (4.5 / 3)^(1/2) = 1.5^(0.5) = 1.2247
    
    phi, J, d = mmphi_intensive(X, q=2, p=2)
    
    assert isinstance(phi, float)
    assert isinstance(J, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert np.isclose(phi, np.sqrt(1.5))
    assert np.array_equal(J, np.array([2, 1]))
    assert np.allclose(d, np.array([np.sqrt(0.5), np.sqrt(2)]))

def test_mmphi_intensive_duplicates():
    """Test that duplicate points are removed."""
    X = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
        [0.0, 0.0]  # Duplicate
    ])
    # Should be treated same as the 3-point plan above
    phi, J, d = mmphi_intensive(X, q=2, p=2)
    
    assert np.isclose(phi, np.sqrt(1.5))
    assert len(d) == 2

def test_mmphi_intensive_insufficient_points():
    """Test with fewer than 2 points."""
    X = np.array([[0.0, 0.0]])
    phi, J, d = mmphi_intensive(X)
    assert phi == np.inf
    
    X_empty = np.array([])
    # The function expects 2D array, let's pass empty 2D
    X_empty_2d = np.zeros((0, 2))
    phi, J, d = mmphi_intensive(X_empty_2d)
    assert phi == np.inf

def test_mmphi_intensive_identical_points_after_unique():
    """
    Test where points are effectively identical or distance is 0.
    However, the function removes duplicates first.
    If we have distinct points that have 0 distance (impossible in metric space unless identical),
    it would be handled by unique.
    Let's test a case where unique leaves 1 point.
    """
    X = np.array([
        [0.5, 0.5],
        [0.5, 0.5]
    ])
    phi, J, d = mmphi_intensive(X)
    assert phi == np.inf

def test_mmphi_intensive_return_types():
    """Verify return types."""
    X = np.random.rand(5, 2)
    phi, J, d = mmphi_intensive(X)
    assert isinstance(phi, float)
    assert isinstance(J, np.ndarray)
    assert isinstance(d, np.ndarray)

def test_mmphi_intensive_update_basic():
    """Test mmphi_intensive_update consistency with mmphi_intensive."""
    # Start with 3 points
    X = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0]
    ])
    q = 2
    p = 2
    
    # Calculate initial state
    phi_initial, J_initial, d_initial = mmphi_intensive(X, q=q, p=p)
    
    # New point to add
    new_point = np.array([0.1, 0.1])
    
    # Update using the update function
    phi_updated, J_updated, d_updated = mmphi_intensive_update(
        X, new_point, J_initial, d_initial, q=q, p=p
    )
    
    # Calculate from scratch with the new point included
    X_new = np.vstack([X, new_point])
    phi_scratch, J_scratch, d_scratch = mmphi_intensive(X_new, q=q, p=p)
    
    # Verify results match
    assert np.isclose(phi_updated, phi_scratch)
    assert np.array_equal(J_updated, J_scratch)
    assert np.allclose(d_updated, d_scratch)

def test_mmphi_intensive_update_consistency():
    """Test consistency across multiple updates."""
    # Start with 2 points
    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0]
    ])
    q = 5
    p = 1
    
    phi, J, d = mmphi_intensive(X, q=q, p=p)
    
    # Add 5 random points sequentially
    np.random.seed(42)
    for _ in range(5):
        new_point = np.random.rand(2)
        
        # Update
        phi_upd, J_upd, d_upd = mmphi_intensive_update(X, new_point, J, d, q=q, p=p)
        
        # Scratch
        X = np.vstack([X, new_point])
        phi_scratch, J_scratch, d_scratch = mmphi_intensive(X, q=q, p=p)
        
        assert np.isclose(phi_upd, phi_scratch)
        assert np.array_equal(J_upd, J_scratch)
        assert np.allclose(d_upd, d_scratch)
        
        # Update state for next iteration
        phi, J, d = phi_upd, J_upd, d_upd
