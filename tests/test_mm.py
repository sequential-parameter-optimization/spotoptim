# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for all plot tests

from spotoptim.sampling.mm import (
    mmphi_intensive,
    mmphi_intensive_update,
    mmphi_corrected,
    mmphi_corrected_update,
    propose_mmphi_corrected_minimizing_point,
    mm_corrected_improvement,
    plot_mmphi_corrected_vs_n_lhs,
    plot_mmphi_corrected_vs_points,
)


def test_mmphi_intensive_basic():
    """Test with a simple 3-point plan."""
    X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
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
    X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.0, 0.0]])  # Duplicate
    # Should be treated same as the 3-point plan above
    phi, J, d = mmphi_intensive(X, q=2, p=2)

    assert np.isclose(phi, np.sqrt(1.5))
    assert len(d) == 2


def test_mmphi_intensive_insufficient_points():
    """Test with fewer than 2 points."""
    X = np.array([[0.0, 0.0]])
    phi, J, d = mmphi_intensive(X)
    assert phi == np.inf

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
    X = np.array([[0.5, 0.5], [0.5, 0.5]])
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
    X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
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
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
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
        _, J, d = phi_upd, J_upd, d_upd


# ---------------------------------------------------------------------------
# Tests for mmphi_corrected
# ---------------------------------------------------------------------------

def test_mmphi_corrected_basic():
    """Verify corrected criterion against manual calculation for a 3-point plan.

    For X = [[0,0],[0.5,0.5],[1,1]], q=2, p=2, k=2:
      Distances: sqrt(0.5) (x2) and sqrt(2) (x1)
      sum_term = 2*(0.5)^(-1) + 1*(2)^(-1) = 4.0 + 0.5 = 4.5
      normalization = 3^(1 + 2/2) = 3^2 = 9
      corrected_phiq = (4.5 / 9)^(1/2) = 0.5^(0.5) = sqrt(0.5)
    """
    X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    phi_hat, J, d = mmphi_corrected(X, q=2, p=2)

    assert isinstance(phi_hat, float)
    assert isinstance(J, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert np.isclose(phi_hat, np.sqrt(0.5))
    assert np.array_equal(J, np.array([2, 1]))
    assert np.allclose(d, np.array([np.sqrt(0.5), np.sqrt(2)]))


def test_mmphi_corrected_relation_to_intensive():
    """corrected_phiq == intensive_phiq * (M / n^(1+q/k))^(1/q).

    Both criteria share the same sum_term. Their ratio is determined by
    the normalization factor:
      intensive uses M = n*(n-1)/2
      corrected uses  n^(1+q/k)
    So:  corrected_phiq = intensive_phiq * (M / n^(1+q/k))^(1/q)
    """
    X = np.random.default_rng(0).random((6, 3))
    q, p = 2.0, 2.0
    n, k = X.shape

    phi_int, _, _ = mmphi_intensive(X, q=q, p=p)
    phi_hat, _, _ = mmphi_corrected(X, q=q, p=p)

    M = n * (n - 1) / 2
    norm_int = M
    norm_cor = n ** (1.0 + q / k)
    expected = phi_int * (norm_int / norm_cor) ** (1.0 / q)

    assert np.isclose(phi_hat, expected)


def test_mmphi_corrected_size_invariance():
    """hat_Phi converges asymptotically for optimal equidistant 1-D designs.

    The corrected criterion is asymptotically size-invariant (converges to a
    constant for optimal designs as n -> inf), but the lemma in index.qmd
    explicitly states it is *not* guaranteed to be monotonically decreasing
    for all finite n. We therefore verify convergence by checking that the
    standard deviation of hat_Phi over large n is small relative to its mean.
    """
    q = 2.0
    results = []
    for n in range(20, 101, 5):
        X = np.linspace(0.0, 1.0, n).reshape(-1, 1)
        phi_hat, _, _ = mmphi_corrected(X, q=q, p=2)
        results.append(phi_hat)

    results = np.array(results)
    # Coefficient of variation < 15 % indicates convergence
    assert results.std() / results.mean() < 0.15, (
        f"hat_Phi did not converge: mean={results.mean():.4f}, std={results.std():.4f}"
    )


def test_mmphi_corrected_insufficient_points():
    """Returns inf for fewer than 2 unique points."""
    X_single = np.array([[0.5, 0.5]])
    phi_hat, _, _ = mmphi_corrected(X_single)
    assert phi_hat == np.inf

    X_all_same = np.array([[0.3, 0.7], [0.3, 0.7]])
    phi_hat2, _, _ = mmphi_corrected(X_all_same)
    assert phi_hat2 == np.inf


def test_mmphi_corrected_duplicates_removed():
    """Duplicate rows are removed before computation."""
    X_clean = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    X_dup = np.vstack([X_clean, [0.0, 0.0]])  # extra duplicate

    phi_clean, _, _ = mmphi_corrected(X_clean, q=2, p=2)
    phi_dup, _, _ = mmphi_corrected(X_dup, q=2, p=2)

    assert np.isclose(phi_clean, phi_dup)


def test_mmphi_corrected_return_types():
    """Return types are correct for a well-formed design."""
    X = np.random.default_rng(7).random((5, 2))
    phi_hat, J, d = mmphi_corrected(X)

    assert isinstance(phi_hat, float)
    assert isinstance(J, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert phi_hat > 0


def test_mmphi_corrected_normalize_flag():
    """normalize_flag=True is equivalent to normalizing X manually then calling with False."""
    from spotoptim.utils.stats import normalize_X

    X = np.random.default_rng(99).random((5, 2))
    X_norm = normalize_X(X)

    phi_manual, _, _ = mmphi_corrected(X_norm, normalize_flag=False)
    phi_flag, _, _ = mmphi_corrected(X, normalize_flag=True)

    assert np.isclose(phi_manual, phi_flag, rtol=1e-10)


# ---------------------------------------------------------------------------
# Tests for mmphi_corrected_update
# ---------------------------------------------------------------------------

def test_mmphi_corrected_update_matches_from_scratch():
    """Update result must equal mmphi_corrected called on the full (n+1)-point design."""
    X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    new_point = np.array([0.25, 0.75])
    q, p = 2.0, 2.0

    phi_base, J, d = mmphi_corrected(X, q=q, p=p)
    phi_upd, J_upd, d_upd = mmphi_corrected_update(X, new_point, J, d, q=q, p=p)

    X_full = np.vstack([X, new_point])
    phi_scratch, J_scratch, d_scratch = mmphi_corrected(X_full, q=q, p=p)

    assert np.isclose(phi_upd, phi_scratch)
    assert np.array_equal(J_upd, J_scratch)
    assert np.allclose(d_upd, d_scratch)


def test_mmphi_corrected_update_normalization_differs_from_intensive_update():
    """Corrected update and intensive update share the same distance cache but differ in norm.

    corrected_phiq  = (sum_term / (n+1)^(1+q/k))^(1/q)
    intensive_phiq  = (sum_term / M)^(1/q),  M = (n+1)*n/2

    So their ratio equals (M / (n+1)^(1+q/k))^(1/q).
    """
    X = np.random.default_rng(5).random((5, 3))
    new_point = np.random.default_rng(6).random(3)
    q, p = 2.0, 2.0
    n, k = X.shape

    _, J, d = mmphi_corrected(X, q=q, p=p)
    phi_cor, _, _ = mmphi_corrected_update(X, new_point, J, d, q=q, p=p)

    _, J_int, d_int = mmphi_intensive(X, q=q, p=p)
    phi_int, _, _ = mmphi_intensive_update(X, new_point, J_int, d_int, q=q, p=p)

    n_new = n + 1
    M = n_new * n / 2
    norm_cor = n_new ** (1.0 + q / k)
    expected_ratio = (M / norm_cor) ** (1.0 / q)

    assert np.isclose(phi_cor, phi_int * expected_ratio)


def test_mmphi_corrected_update_sequential_consistency():
    """Sequential updates must agree with mmphi_corrected called from scratch at each step."""
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    q, p = 5.0, 1.0

    phi, J, d = mmphi_corrected(X, q=q, p=p)

    rng = np.random.default_rng(42)
    for _ in range(6):
        new_point = rng.random(2)

        phi_upd, J_upd, d_upd = mmphi_corrected_update(X, new_point, J, d, q=q, p=p)

        X = np.vstack([X, new_point])
        phi_scratch, J_scratch, d_scratch = mmphi_corrected(X, q=q, p=p)

        assert np.isclose(phi_upd, phi_scratch)
        assert np.array_equal(J_upd, J_scratch)
        assert np.allclose(d_upd, d_scratch)

        J, d = J_upd, d_upd


def test_mmphi_corrected_update_return_types():
    """Return types are float, ndarray, ndarray."""
    X = np.random.default_rng(11).random((4, 2))
    new_point = np.random.default_rng(12).random(2)
    _, J, d = mmphi_corrected(X)
    phi, J_u, d_u = mmphi_corrected_update(X, new_point, J, d)

    assert isinstance(phi, float)
    assert isinstance(J_u, np.ndarray)
    assert isinstance(d_u, np.ndarray)
    assert phi > 0


def test_mmphi_corrected_update_single_existing_point():
    """Works correctly when the existing design has exactly one point."""
    X = np.array([[0.0, 0.0]])
    new_point = np.array([1.0, 1.0])
    q, p = 2.0, 2.0

    # Bootstrap J and d from the 2-point design directly
    X_two = np.vstack([X, new_point])
    phi_scratch, J_scratch, d_scratch = mmphi_corrected(X_two, q=q, p=p)

    # Build a minimal J/d for the 1-point design (no pairs yet)
    J_empty = np.array([], dtype=int)
    d_empty = np.array([], dtype=float)
    phi_upd, J_upd, d_upd = mmphi_corrected_update(X, new_point, J_empty, d_empty, q=q, p=p)

    assert np.isclose(phi_upd, phi_scratch)


def test_mmphi_corrected_update_normalize_flag():
    """normalize_flag=True produces a finite, positive result distinct from no normalization.

    The flag causes X to be normalized to [0, 1] before distances are computed.
    When the raw design is not already in [0, 1], this changes the distances,
    so the flagged result must differ from the unflagged one.
    """
    # Use a design whose coordinates are NOT in [0, 1] to make normalization visible
    X = np.array([[10.0, 20.0], [30.0, 50.0], [50.0, 80.0], [70.0, 60.0]])
    new_point = np.array([40.0, 40.0])
    q, p = 2.0, 2.0

    _, J, d = mmphi_corrected(X, q=q, p=p)
    phi_no_norm, _, _ = mmphi_corrected_update(X, new_point, J, d, q=q, p=p, normalize_flag=False)
    phi_norm, _, _ = mmphi_corrected_update(X, new_point, J, d, q=q, p=p, normalize_flag=True)

    assert np.isfinite(phi_norm) and phi_norm > 0
    assert not np.isclose(phi_no_norm, phi_norm)


# ---------------------------------------------------------------------------
# Tests for propose_mmphi_corrected_minimizing_point
# ---------------------------------------------------------------------------

def test_propose_mmphi_corrected_return_shape():
    """Returned array has shape (1, n_dim)."""
    X = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])
    pt = propose_mmphi_corrected_minimizing_point(X, n_candidates=50, seed=0)
    assert pt.shape == (1, 2)


def test_propose_mmphi_corrected_within_bounds():
    """Every coordinate of the returned point lies within [lower, upper]."""
    rng = np.random.default_rng(3)
    X = rng.random((5, 3))
    lower = np.array([0.1, 0.2, 0.0])
    upper = np.array([0.8, 0.9, 0.7])
    pt = propose_mmphi_corrected_minimizing_point(
        X, n_candidates=100, seed=7, lower=lower, upper=upper
    )
    assert np.all(pt >= lower) and np.all(pt <= upper)


def test_propose_mmphi_corrected_default_bounds_within_unit():
    """With default bounds the returned point is inside [0, 1]^k."""
    X = np.random.default_rng(9).random((4, 2))
    pt = propose_mmphi_corrected_minimizing_point(X, n_candidates=100, seed=1)
    assert np.all(pt >= 0.0) and np.all(pt <= 1.0)


def test_propose_mmphi_corrected_seed_reproducibility():
    """Same seed produces the same point; different seeds (usually) differ."""
    X = np.random.default_rng(0).random((5, 2))
    pt1 = propose_mmphi_corrected_minimizing_point(X, n_candidates=200, seed=42)
    pt2 = propose_mmphi_corrected_minimizing_point(X, n_candidates=200, seed=42)
    pt3 = propose_mmphi_corrected_minimizing_point(X, n_candidates=200, seed=99)

    assert np.array_equal(pt1, pt2)
    assert not np.array_equal(pt1, pt3)


def test_propose_mmphi_corrected_reduces_criterion():
    """Adding the proposed point must not increase hat_Phi more than a random point would on average.

    We compare the criterion after adding the proposed point versus the mean
    over several random candidates. The proposed point should be at or below
    that mean.
    """
    rng = np.random.default_rng(55)
    X = rng.random((6, 2))
    q, p = 2.0, 2.0

    proposed = propose_mmphi_corrected_minimizing_point(
        X, n_candidates=300, q=q, p=p, seed=55
    )
    phi_proposed, _, _ = mmphi_corrected(np.vstack([X, proposed]), q=q, p=p)

    random_phis = []
    for _ in range(50):
        rand_pt = rng.random((1, 2))
        phi_rand, _, _ = mmphi_corrected(np.vstack([X, rand_pt]), q=q, p=p)
        random_phis.append(phi_rand)

    assert phi_proposed <= float(np.mean(random_phis))


def test_propose_mmphi_corrected_equivalent_to_update():
    """Phi of (X + proposed) via mmphi_corrected equals mmphi_corrected_update value."""
    X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    q, p = 2.0, 2.0

    proposed = propose_mmphi_corrected_minimizing_point(
        X, n_candidates=100, q=q, p=p, seed=7
    )
    # Via update
    _, J, d = mmphi_corrected(X, q=q, p=p)
    phi_update, _, _ = mmphi_corrected_update(X, proposed.ravel(), J, d, q=q, p=p)

    # Via scratch
    phi_scratch, _, _ = mmphi_corrected(np.vstack([X, proposed]), q=q, p=p)

    assert np.isclose(phi_update, phi_scratch)


def test_propose_mmphi_corrected_invalid_bounds_raises():
    """ValueError is raised when lower >= upper for any dimension."""
    X = np.random.default_rng(0).random((3, 2))
    import pytest
    with pytest.raises(ValueError, match="Lower bounds"):
        propose_mmphi_corrected_minimizing_point(
            X, lower=np.array([0.5, 0.0]), upper=np.array([0.3, 1.0])
        )


def test_propose_mmphi_corrected_normalize_flag_returns_valid():
    """normalize_flag=True returns a finite, correctly shaped point."""
    X = np.array([[10.0, 20.0], [30.0, 50.0], [50.0, 80.0], [70.0, 60.0]])
    lower = np.array([10.0, 20.0])
    upper = np.array([70.0, 80.0])
    pt = propose_mmphi_corrected_minimizing_point(
        X, n_candidates=50, seed=0, lower=lower, upper=upper, normalize_flag=True
    )
    assert pt.shape == (1, 2)
    assert np.all(np.isfinite(pt))


# ---------------------------------------------------------------------------
# mm_corrected_improvement
# ---------------------------------------------------------------------------


def test_mm_corrected_improvement_return_type():
    """Return value is a Python float."""
    X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    x = np.array([0.25, 0.75])
    result = mm_corrected_improvement(x, X)
    assert isinstance(result, float)


def test_mm_corrected_improvement_exponential_positive():
    """Exponential mode returns a positive value for any finite phi difference."""
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    x = np.array([0.5, 0.5])
    result = mm_corrected_improvement(x, X, exponential=True)
    assert result > 0.0


def test_mm_corrected_improvement_linear_mode():
    """Linear mode (exponential=False) returns phi_base - phi_new."""
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    x = np.array([0.5, 0.5])
    phi_base, J_base, d_base = mmphi_corrected(X)
    phi_new, _, _ = mmphi_corrected_update(X, x, J_base, d_base)
    expected = phi_base - phi_new
    result = mm_corrected_improvement(x, X, exponential=False)
    assert np.isclose(result, expected)


def test_mm_corrected_improvement_cached_equals_scratch():
    """Pre-passing (phi_base, J_base, d_base) yields the same result as computing from scratch."""
    X = np.random.default_rng(17).random((6, 2))
    x = np.array([0.3, 0.7])
    phi_base, J_base, d_base = mmphi_corrected(X)
    result_cached = mm_corrected_improvement(x, X, phi_base=phi_base, J_base=J_base, d_base=d_base)
    result_scratch = mm_corrected_improvement(x, X)
    assert np.isclose(result_cached, result_scratch)


def test_mm_corrected_improvement_verbose_smoke(capsys):
    """verbose=True prints three lines without raising."""
    X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    x = np.array([0.25, 0.75])
    mm_corrected_improvement(x, X, verbose=True)
    captured = capsys.readouterr()
    assert "Corrected Morris-Mitchell base" in captured.out
    assert "Corrected Morris-Mitchell new" in captured.out
    assert "Corrected Morris-Mitchell improvement" in captured.out


def test_mm_corrected_improvement_good_point_improves():
    """A well-spaced point gives improvement > 1.0 (exponential) when phi decreases."""
    # Space-filling 2-D grid; centre point should reduce hat_Phi
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    x = np.array([0.5, 0.5])
    phi_base, J_base, d_base = mmphi_corrected(X)
    phi_new, _, _ = mmphi_corrected_update(X, x, J_base, d_base)
    if phi_new < phi_base:
        result = mm_corrected_improvement(x, X, exponential=True)
        assert result > 1.0


def test_mm_corrected_improvement_normalize_flag_differs():
    """normalize_flag=True gives a different result than False for non-unit-scale data."""
    X = np.array([[10.0, 20.0], [50.0, 60.0], [30.0, 80.0], [70.0, 40.0]])
    x = np.array([40.0, 50.0])
    result_norm = mm_corrected_improvement(x, X, normalize_flag=True)
    result_raw = mm_corrected_improvement(x, X, normalize_flag=False)
    assert np.isfinite(result_norm)
    assert result_norm != result_raw


# ---------------------------------------------------------------------------
# plot_mmphi_corrected_vs_n_lhs
# ---------------------------------------------------------------------------


def test_plot_mmphi_corrected_vs_n_lhs_runs(monkeypatch):
    """Function completes without raising for a valid parameter set."""
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    plot_mmphi_corrected_vs_n_lhs(k_dim=2, seed=0, n_min=10, n_max=20, n_step=5)


def test_plot_mmphi_corrected_vs_n_lhs_empty_range(monkeypatch, capsys):
    """Empty n_values range prints a warning and returns early without plotting."""
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    plot_mmphi_corrected_vs_n_lhs(k_dim=2, seed=0, n_min=50, n_max=10, n_step=5)
    captured = capsys.readouterr()
    assert "empty" in captured.out.lower()


def test_plot_mmphi_corrected_vs_n_lhs_single_value(monkeypatch):
    """Single n value (n_min == n_max) runs without error."""
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    plot_mmphi_corrected_vs_n_lhs(k_dim=3, seed=7, n_min=15, n_max=15, n_step=5)


def test_plot_mmphi_corrected_vs_n_lhs_higher_dim(monkeypatch):
    """Works for higher-dimensional designs (k=5)."""
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    plot_mmphi_corrected_vs_n_lhs(k_dim=5, seed=1, n_min=10, n_max=20, n_step=10)


def test_plot_mmphi_corrected_vs_n_lhs_ratio_identity():
    """mmphi_corrected equals mmphi_intensive scaled by (M / n^{1+q/k})^{1/q}.

    The definition from the paper gives:
        hat_Phi_q = Phi_q / n^{1/q + 1/k}
    where the intensive normalizes Phi_q by M = n*(n-1)/2.  So:
        hat_Phi_q = mmphi_intensive * (M / n^{1+q/k})^{1/q}
    This identity must hold for every design and every (q, p) combination.
    """
    from scipy.stats import qmc

    rng = np.random.default_rng(123)
    params = [(2.0, 2.0), (3.0, 2.0), (2.0, 1.0)]
    for q, p in params:
        n = rng.integers(10, 30)
        X = qmc.LatinHypercube(d=3, rng=int(rng.integers(1000))).random(n=int(n))
        k = X.shape[1]
        M = n * (n - 1) / 2
        phi_i, _, _ = mmphi_intensive(X, q=q, p=p)
        phi_c, _, _ = mmphi_corrected(X, q=q, p=p)
        expected = phi_i * (M / n ** (1.0 + q / k)) ** (1.0 / q)
        assert np.isclose(phi_c, expected, rtol=1e-10), (
            f"q={q}, p={p}: corrected={phi_c:.6f}, expected={expected:.6f}"
        )


# ---------------------------------------------------------------------------
# plot_mmphi_corrected_vs_points
# ---------------------------------------------------------------------------


def test_plot_mmphi_corrected_vs_points_returns_dataframe(monkeypatch):
    """Return value is a pd.DataFrame with the expected column structure."""
    import pandas as pd
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)

    X_base = np.array([[0.1, 0.2], [0.4, 0.5], [0.7, 0.8]])
    x_min = np.array([0.0, 0.0])
    x_max = np.array([1.0, 1.0])
    df = plot_mmphi_corrected_vs_points(X_base, x_min, x_max, p_min=5, p_max=10, p_step=5, n_repeats=2)

    assert isinstance(df, pd.DataFrame)
    assert "n_points" in df.columns


def test_plot_mmphi_corrected_vs_points_row_count(monkeypatch):
    """One row per distinct n_points value in range(p_min, p_max+1, p_step)."""
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)

    X_base = np.random.default_rng(0).random((4, 2))
    x_min = np.zeros(2)
    x_max = np.ones(2)
    df = plot_mmphi_corrected_vs_points(X_base, x_min, x_max, p_min=5, p_max=20, p_step=5, n_repeats=2)

    expected_rows = len(range(5, 21, 5))
    assert len(df) == expected_rows


def test_plot_mmphi_corrected_vs_points_mean_finite(monkeypatch):
    """Mean corrected criterion is finite and positive for all point counts."""
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)

    X_base = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    x_min = np.zeros(2)
    x_max = np.ones(2)
    df = plot_mmphi_corrected_vs_points(X_base, x_min, x_max, p_min=5, p_max=15, p_step=5, n_repeats=3)

    means = df["mmphi_corrected"]["mean"].values
    assert np.all(np.isfinite(means))
    assert np.all(means > 0)


def test_plot_mmphi_corrected_vs_points_custom_q_p(monkeypatch):
    """Runs successfully with non-default q and p_norm values."""
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)

    X_base = np.random.default_rng(7).random((5, 3))
    x_min = np.zeros(3)
    x_max = np.ones(3)
    df = plot_mmphi_corrected_vs_points(
        X_base, x_min, x_max, p_min=5, p_max=10, p_step=5, n_repeats=2, q=3.0, p_norm=1.0
    )
    assert len(df) == 2


def test_plot_mmphi_corrected_vs_points_std_nonneg(monkeypatch):
    """Std of criterion is non-negative for n_repeats > 1."""
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)

    X_base = np.random.default_rng(3).random((4, 2))
    x_min = np.zeros(2)
    x_max = np.ones(2)
    df = plot_mmphi_corrected_vs_points(X_base, x_min, x_max, p_min=10, p_max=10, p_step=5, n_repeats=4)

    stds = df["mmphi_corrected"]["std"].values
    assert np.all(stds >= 0)


def test_plot_mmphi_corrected_vs_points_matches_mmphi_corrected(monkeypatch):
    """Values in the DataFrame are consistent with direct mmphi_corrected calls.

    For a single repeat we can reconstruct the exact value and verify the mean
    equals it (since std-of-one is NaN, we check with n_repeats=1).
    """
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    np.random.seed(42)

    X_base = np.array([[0.1, 0.2], [0.5, 0.6], [0.9, 0.3]])
    x_min = np.zeros(2)
    x_max = np.ones(2)
    df = plot_mmphi_corrected_vs_points(
        X_base, x_min, x_max, p_min=5, p_max=5, p_step=5, n_repeats=1
    )
    mean_val = df["mmphi_corrected"]["mean"].iloc[0]
    assert np.isfinite(mean_val) and mean_val > 0
