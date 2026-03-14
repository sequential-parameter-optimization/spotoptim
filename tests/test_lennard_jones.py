# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Lennard-Jones function."""

import numpy as np
import pytest
from spotoptim.function.so import lennard_jones


class TestLennardJones:
    """Test suite for the Lennard-Jones function."""

    def test_output_shape_single(self):
        """Test output shape for single 1D input."""
        n_atoms = 13
        # 3 coords per atom
        X = np.random.rand(3 * n_atoms)
        result = lennard_jones(X, n_atoms=n_atoms)
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_output_shape_batch(self):
        """Test output shape for batch input."""
        n_atoms = 13
        n_samples = 5
        X = np.random.rand(n_samples, 3 * n_atoms)
        result = lennard_jones(X, n_atoms=n_atoms)
        assert result.shape == (n_samples,)
        assert np.all(np.isfinite(result))

    def test_input_dimension_validation(self):
        """Test that invalid dimensions raise ValueError."""
        n_atoms = 13
        X = np.random.rand(1, 3 * n_atoms + 1)  # Incorrect size
        # Escape * for regex match or use a simpler substring
        with pytest.raises(ValueError, match=r"Input dimension must be 3 \* n_atoms"):
            lennard_jones(X, n_atoms=n_atoms)

    def test_variable_atom_count(self):
        """Test with different number of atoms."""
        n_atoms = 4
        X = np.random.rand(1, 3 * n_atoms)
        result = lennard_jones(X, n_atoms=n_atoms)
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_minimum_property(self):
        """Test that the global minimum is reasonable (sanity check).

        We don't expect to find the exact global minimum easily,
        but we can check if a slightly perturbed minimum has higher energy.
        """
        # Approximate global minimum for N=13
        # In [0,1] input space, 0.5 maps to 0.0 in [-2, 2] space.
        # If all are at 0.5, they all overlap -> distance 0 -> clipped -> huge repulsion

        X_overlap = np.ones((1, 39)) * 0.5
        E_overlap = lennard_jones(X_overlap)

        # Random dispersed configuration should have lower energy than overlapping
        X_random = np.random.rand(1, 39)
        E_random = lennard_jones(X_random)

        assert (
            E_overlap > E_random
        ), "Overlapping atoms should have huge repulsion energy"

    def test_deterministic(self):
        """Test that function is deterministic."""
        X = np.random.rand(5, 39)
        val1 = lennard_jones(X)
        val2 = lennard_jones(X)
        np.testing.assert_array_equal(val1, val2)

    def test_scaling_check(self):
        """Confirm that inputs in [0, 1] are actually mapped to larger range.

        If we input 0 and 1, distance in scaled space should be significant.
        """
        # Atom 1 at [0,0,0] (in input) -> [-2,-2,-2]
        # Atom 2 at [1,1,1] (in input) -> [2,2,2]
        # Distance should be sqrt(4^2 + 4^2 + 4^2) = sqrt(48) approx 6.9

        X = np.zeros((1, 6))
        X[0, 0:3] = 0.0  # Atom 1
        X[0, 3:6] = 1.0  # Atom 2

        # We can't access internal coords, but we can verify E is small (attractive/zero) rather than huge repulsion
        # At r=6.9, potential is very close to 0 (as r >> 1).

        val = lennard_jones(X, n_atoms=2)
        assert (
            abs(val[0]) < 1.0
        ), f"At large separation, energy should be near 0, got {val[0]}"

        # Now place them closer in input space
        # To test overlapping, force both to SAME input value.
        X[0, 3:6] = 0.0  # Both at 0.0
        val_close = lennard_jones(X, n_atoms=2)
        assert val_close[0] > 1000, "Overlapping atoms should repulse strongly"

    def test_pair_interaction_explicit(self):
        """Test explicitly a 2-atom system with known distance."""
        # Scaled space: r_min for LJ is 2^(1/6) approx 1.122
        # Potential at minimum should be -epsilon = -1.
        # Total energy = 4 * (-1) = -4? Wait, formula is 4*eps*(...).
        # At r_min, term = -0.25 (since 4*term = -1 => term = -0.25).
        # Actually standard LJ form: 4*((1/r)^12 - (1/r)^6). Min is -1 at r = 2^(1/6).

        # Let's construct input such that distance is 2^(1/6).
        # Map: x_scaled = x_in * 4 - 2
        # We want x_scaled_2 - x_scaled_1 = 2^(1/6)
        # (x_in_2 - x_in_1) * 4 = 2^(1/6)
        # x_in_delta = 2^(1/6) / 4 approx 0.2806

        r_min = 2 ** (1 / 6)
        delta_in = r_min / 4.0

        # Align along X axis
        X = np.zeros((1, 6)) + 0.5  # Center everyone at 0
        X[0, 0] = 0.5  # Atom 1 at 0
        X[0, 3] = 0.5 + delta_in  # Atom 2 at r_min

        # Energy should be exactly -1.0
        val = lennard_jones(X, n_atoms=2)

        # Allow small floating point error
        assert np.isclose(
            val[0], -1.0, atol=1e-4
        ), f"Expected -1.0 at equilibrium, got {val[0]}"
