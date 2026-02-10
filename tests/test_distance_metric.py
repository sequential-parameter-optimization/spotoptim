# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pytest
from spotoptim import SpotOptim
from scipy.spatial.distance import cdist

# List of standard metrics supported by cdist that work with default arguments
# Note: 'seuclidean' and 'mahalanobis' usually require V/VI arguments or imply sample variance
# 'minkowski' defaults to p=2
metrics = [
    "braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine",
    "euclidean", "jensenshannon", "minkowski", "sqeuclidean"
]

@pytest.mark.parametrize("metric", metrics)
def test_select_new_metrics(metric):
    """Test select_new with various distance metrics."""
    # 2D case
    # Use vectors that are:
    # 1. Non-constant (for correlation)
    # 2. Not collinear (for cosine)
    # 3. Not proportional (for jensenshannon)
    X = np.array([[0.1, 0.2], [0.5, 0.6]])
    
    # Candidate identical to first point
    A_dup = np.array([[0.1, 0.2]])
    
    # Candidate far away and distinct in direction/distribution
    # Point 1: [0.1, 0.2] -> direction [1, 2], dist [1/3, 2/3]
    # Point 2: [0.5, 0.6] -> direction [5, 6], dist [5/11, 6/11]
    # New: [0.8, 0.1] -> direction [8, 1], dist [8/9, 1/9]
    A_new = np.array([[0.8, 0.1]])
    
    opt = SpotOptim(fun=lambda x: np.sum(x**2), bounds=[(0, 1), (0, 1)], min_tol_metric=metric)
    
    # 1. Test duplicate detection
    # Distance should be 0 (or very close to 0 due to float errors)
    # Tolerance small but > 0
    tol = 1e-5
    
    # Should be detected as duplicate
    new_A, is_new = opt.select_new(A_dup, X, tolerance=tol)
    assert len(new_A) == 0
    assert is_new[0] == False
    
    # 2. Test new point acceptance
    # Should be accepted
    new_A, is_new = opt.select_new(A_new, X, tolerance=tol)
    assert len(new_A) == 1
    assert is_new[0] == True

def test_select_new_chebyshev_corner_case():
    """Specific test for Chebyshev behavior vs Euclidean."""
    X = np.array([[0.0, 0.0]])
    tol = 1.0
    
    # Point at corner (0.9, 0.9)
    # Chebyshev dist = 0.9 <= 1.0 -> Duplicate
    # Euclidean dist = 1.27 > 1.0 -> New
    A_corner = np.array([[0.9, 0.9]])
    
    # Chebyshev
    opt_cheb = SpotOptim(fun=lambda x: np.sum(x**2), bounds=[(-5, 5), (-5, 5)], min_tol_metric="chebyshev")
    _, is_new_cheb = opt_cheb.select_new(A_corner, X, tolerance=tol)
    assert is_new_cheb[0] == False
    
    # Euclidean
    opt_eucl = SpotOptim(fun=lambda x: np.sum(x**2), bounds=[(-5, 5), (-5, 5)], min_tol_metric="euclidean")
    _, is_new_eucl = opt_eucl.select_new(A_corner, X, tolerance=tol)
    assert is_new_eucl[0] == True

def test_metric_defaults():
    """Test that default metric is chebyshev."""
    opt = SpotOptim(fun=lambda x: np.sum(x**2), bounds=[(-5, 5)])
    assert opt.min_tol_metric == "chebyshev"

