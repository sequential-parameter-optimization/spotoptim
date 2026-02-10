# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import numpy as np
from spotoptim.utils.stats import get_sample_size

def test_get_sample_size_basic():
    """Test standard case from documentation."""
    alpha = 0.05
    beta = 0.2
    sigma = 1.0
    delta = 1.0
    
    n = get_sample_size(alpha, beta, sigma, delta)
    
    # Expected value approx 15.6978
    assert np.isclose(n, 15.6978, atol=1e-4)

def test_get_sample_size_sigma_scale():
    """Test that n scales with sigma squared."""
    alpha = 0.05
    beta = 0.2
    sigma1 = 1.0
    delta = 1.0
    n1 = get_sample_size(alpha, beta, sigma1, delta)
    
    sigma2 = 2.0
    n2 = get_sample_size(alpha, beta, sigma2, delta)
    
    # n depends on sigma^2, so doubling sigma should quadruple n
    assert np.isclose(n2, 4 * n1)

def test_get_sample_size_delta_scale():
    """Test that n scales inversely with delta squared."""
    alpha = 0.05
    beta = 0.2
    sigma = 1.0
    delta1 = 1.0
    n1 = get_sample_size(alpha, beta, sigma, delta1)
    
    delta2 = 0.5
    n2 = get_sample_size(alpha, beta, sigma, delta2)
    
    # Halving delta should quadruple n
    assert np.isclose(n2, 4 * n1)
