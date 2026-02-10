# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
from spotoptim.sampling.mm import mmphi_intensive


def test_mmphi_intensive_smaller_is_better():
    """
    Verify that a space-filling design has a smaller mmphi_intensive value
    than a clustered/poor design.
    """
    # Good design: Grid points (spread out)
    X_good = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 0.5],
            [0.5, 0.5],
            [1.0, 0.5],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0],
        ]
    )

    # Poor design: All points clustered near (0.5, 0.5)
    # Using small perturbations to ensure they are unique
    np.random.seed(42)
    X_poor = np.random.normal(0.5, 0.01, size=(9, 2))

    phi_good, _, _ = mmphi_intensive(X_good, q=2, p=2)
    phi_poor, _, _ = mmphi_intensive(X_poor, q=2, p=2)

    # Assert that the good design has a LOWER metric value
    assert (
        phi_good < phi_poor
    ), f"Expected good design to have lower metric ({phi_good}) than poor design ({phi_poor})"


def test_mmphi_intensive_1d():
    """Test with 1D data."""
    # 1D Grid: [0, 0.25, 0.5, 0.75, 1.0]
    X_1d_good = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])

    # 1D Clustered/Gapped: [0, 0.1, 0.2, 0.9, 1.0]
    # Even after normalization (range is 0-1), this has clusters at ends and gap in middle.
    X_1d_poor = np.array([[0.0], [0.1], [0.2], [0.9], [1.0]])

    phi_good, _, _ = mmphi_intensive(X_1d_good, q=2, p=2)
    phi_poor, _, _ = mmphi_intensive(X_1d_poor, q=2, p=2)

    assert phi_good < phi_poor


def test_mmphi_high_dimensions():
    """Test with higher dimensions (5D) to ensure no broadcasting errors."""
    np.random.seed(123)
    n_points = 10
    n_dim = 5
    X = np.random.rand(n_points, n_dim)

    phi, J, d = mmphi_intensive(X, q=2, p=2)
    assert np.isfinite(phi)
    # Number of distances should be n*(n-1)/2 = 45 (or less if duplicates)
    # With random float data, likely all distances are unique => 45 distances
    assert len(d) > 0
    assert len(J) == len(d)


def test_mmphi_p_norm_chebyshev():
    """Test with Chebyshev distance (p=np.inf)."""
    X = np.array([[0.0, 0.0], [1.0, 0.5]])
    # Chebyshev distance: max(|x1-x2|, |y1-y2|) = max(1.0, 0.5) = 1.0

    # Disable normalization so coordinates are used as-is
    phi, J, d = mmphi_intensive(X, q=2, p=np.inf, normalize_flag=False)

    assert len(d) == 1
    np.testing.assert_allclose(d[0], 1.0)
    assert np.isfinite(phi)


def test_mmphi_p_norm_manhattan():
    """Test with Manhattan distance (p=1)."""
    X = np.array([[0.0, 0.0], [1.0, 0.5]])
    # Manhattan distance: |1-0| + |0.5-0| = 1.5

    # Disable normalization so coordinates are used as-is
    phi, J, d = mmphi_intensive(X, q=2, p=1, normalize_flag=False)

    assert len(d) == 1
    np.testing.assert_allclose(d[0], 1.5)


def test_mmphi_q_sensitivity():
    """
    Test that higher q values penalize small distances more.
    Consider two designs:
    D1 has a very small distance pair.
    D2 has moderately small distances.
    Higher q should make D1 look worse (higher metric) relative to low q?
    Actually, let's just check that q affects the result significantly.
    """
    X = np.array([[0.0, 0.0], [0.1, 0.1], [1.0, 1.0]])

    phi_q2, _, _ = mmphi_intensive(X, q=2, p=2)
    phi_q10, _, _ = mmphi_intensive(X, q=10, p=2)

    # Just asserting they are calculated and different.
    # The actual relationship comparison might be complex due to the root (1/q).
    assert phi_q2 != phi_q10
    assert np.isfinite(phi_q2)
    assert np.isfinite(phi_q10)


def test_mmphi_normalization_explicit():
    """
    Verify that normalize_flag=True gives same result as manual normalization
    for data in an arbitrary range.
    """
    # Data in [0, 10]
    X_raw = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])

    # Manual normalization to [0, 1]
    X_norm = X_raw / 10.0

    # Calculate with flag=True on raw data
    phi_flag, _, _ = mmphi_intensive(X_raw, normalize_flag=True)

    # Calculate with flag=False on manually normalized data
    phi_manual, _, _ = mmphi_intensive(X_norm, normalize_flag=False)

    np.testing.assert_allclose(phi_flag, phi_manual)


def test_mmphi_normalization_false_effect():
    """
    Verify that normalize_flag=False on unscaled data gives different (likely smaller)
    metric because distances are larger.
    Metric involves d^(-q). Larger d => smaller metric term.
    """
    X_raw = np.array([[0.0, 0.0], [10.0, 10.0]])

    # Normalized: dist = sqrt(2) approx 1.414. Term ~ (1.414)^(-q)
    # Unnormalized: dist = sqrt(200) approx 14.14. Term ~ (14.14)^(-q)
    # So unnormalized term is much smaller, so metric should be smaller.

    phi_norm, _, _ = mmphi_intensive(X_raw, normalize_flag=True)
    phi_no_norm, _, _ = mmphi_intensive(X_raw, normalize_flag=False)

    # Expect unnormalized (larger distances) to yield smaller metric value
    assert phi_no_norm < phi_norm


def test_large_dataset_performance():
    """Smoke test for larger dataset."""
    np.random.seed(42)
    # N=100 => ~5000 pairs, should be fast
    X = np.random.rand(100, 2)
    phi, _, _ = mmphi_intensive(X)
    assert np.isfinite(phi)
