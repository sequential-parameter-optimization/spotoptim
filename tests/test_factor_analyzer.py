# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import numpy as np
import pandas as pd
from spotoptim.factor_analyzer import (
    FactorAnalyzer,
    calculate_bartlett_sphericity,
    calculate_kmo,
)

@pytest.fixture
def dummy_data():
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    # Create correlated data
    X = np.random.randn(n_samples, n_features)
    # Make some columns correlated
    X[:, 0] = X[:, 1] + np.random.normal(0, 0.1, n_samples)
    X[:, 2] = X[:, 3] + np.random.normal(0, 0.1, n_samples)
    return pd.DataFrame(X, columns=[f"V{i}" for i in range(n_features)])

def test_factor_analyzer_init():
    fa = FactorAnalyzer(n_factors=3, rotation="varimax")
    assert fa.n_factors == 3
    assert fa.rotation == "varimax"

def test_factor_analyzer_fit(dummy_data):
    fa = FactorAnalyzer(n_factors=2, rotation="varimax")
    fa.fit(dummy_data)
    assert fa.loadings_ is not None
    assert fa.loadings_.shape == (10, 2)
    assert fa.get_eigenvalues() is not None

def test_factor_analyzer_transform(dummy_data):
    fa = FactorAnalyzer(n_factors=2, rotation="varimax")
    fa.fit(dummy_data)
    X_transformed = fa.transform(dummy_data)
    assert X_transformed.shape == (100, 2)

def test_bartlett_sphericity(dummy_data):
    """Test Bartlett's sphericity check."""
    chi_square_value, p_value = calculate_bartlett_sphericity(dummy_data)
    assert not np.isnan(chi_square_value)
    assert not np.isnan(p_value)

def test_kmo(dummy_data):
    """Test KMO calculation."""
    kmo_all, kmo_model = calculate_kmo(dummy_data)
    assert not np.isnan(kmo_model)
    assert 0 <= kmo_model <= 1
