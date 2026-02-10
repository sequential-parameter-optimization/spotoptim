# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for tricands integration in SpotOptim.
"""

import numpy as np
from unittest.mock import MagicMock, patch
from spotoptim.SpotOptim import SpotOptim


class TestTricandsIntegration:
    """Test suite for tricands acquisition optimizer integration."""

    def setup_method(self):
        """Setup basic optimizer params."""
        self.bounds = [(-5, 5), (0, 10)]
        self.fun = lambda x: np.sum(x**2)
        self.opt = SpotOptim(
            fun=self.fun,
            bounds=self.bounds,
            acquisition_optimizer="tricands",
            n_initial=5,
            acquisition_fun_return_size=3,
            seed=42,
        )
        # Mock surrogate fitting to avoid overhead
        self.opt.model = MagicMock()
        self.opt.model.predict.return_value = (np.zeros(10), np.zeros(10))

    def test_fallback_when_no_points(self):
        """Test fallback to random sampling when not enough points."""
        # Ensure X_ is None or empty
        self.opt.X_ = None

        candidates = self.opt.optimize_acquisition_func()

        # Should return 'acquisition_fun_return_size' points
        assert candidates.shape == (3, 2)
        # Check bounds
        assert np.all(candidates[:, 0] >= -5)
        assert np.all(candidates[:, 0] <= 5)
        assert np.all(candidates[:, 1] >= 0)
        assert np.all(candidates[:, 1] <= 10)

    def test_tricands_execution(self):
        """Test standard execution with enough points."""
        # Mock X_ with enough points for triangulation (n > m+1)
        self.opt.X_ = np.array([[-4, 1], [-2, 8], [0, 5], [2, 2], [4, 9], [1, 1]])

        # Mock acquisition function to return predictable values
        # We want to verify that the BEST candidates are chosen.
        # _acquisition_function returns NEGATIVE values.
        # Lets mock it to return values such that specific indices are chosen.

        def side_effect(x):
            # Return a value based on the candidate's coordinates to ensure deterministic selection.
            # We want point 0 ([-5, 0]) to have lowest value (best), then point 1, etc.
            # Let's just sum the coordinates.
            # x is shape (2,)
            return np.sum(x)

        self.opt._acquisition_function = MagicMock(side_effect=side_effect)

        with patch("spotoptim.SpotOptim.tricands") as mock_tricands:
            # Mock tricands returning 10 candidates
            # We explicitly control values to ensure deterministic sorting order.
            mock_cands_norm = np.full(
                (10, 2), 0.99
            )  # Initialize with high values (worst)

            # Set specific best candidates (lowest sums)
            mock_cands_norm[0] = [0.0, 0.0]  # Sum -> Lowest
            mock_cands_norm[1] = [0.1, 0.1]  # Sum -> Second Lowest
            mock_cands_norm[2] = [0.2, 0.2]  # Sum -> Third Lowest

            mock_tricands.return_value = mock_cands_norm

            candidates = self.opt.optimize_acquisition_func()

            # Verify tricands called with normalized data
            mock_tricands.assert_called_once()
            args, kwargs = mock_tricands.call_args
            # Check input X was normalized
            # Start of X_ is [-4, 1]. In bounds [-5, 5], [0, 10]:
            # x1 norm: (-4 - -5)/10 = 0.1
            # x2 norm: (1 - 0)/10 = 0.1
            np.testing.assert_allclose(args[0][0], [0.1, 0.1])
            assert kwargs["nmax"] >= 100 * 2

            # Verify denormalization of result
            # Expected denormalized values:
            # val = norm * (upper - lower) + lower
            # ranges: x1: 10, lower -5. x2: 10, lower 0.

            # Cand 0: [0, 0] -> [-5, 0]
            expected_0 = np.array([-5.0, 0.0])

            # Cand 1: [0.1, 0.1]
            # x1 = 0.1*10 - 5 = -4.0
            # x2 = 0.1*10 + 0 = 1.0
            expected_1 = np.array([-4.0, 1.0])

            # Cand 2: [0.2, 0.2]
            # x1 = 0.2*10 - 5 = -3.0
            # x2 = 0.2*10 + 0 = 2.0
            expected_2 = np.array([-3.0, 2.0])

            assert len(candidates) == 3
            np.testing.assert_allclose(candidates[0], expected_0)
            np.testing.assert_allclose(candidates[1], expected_1)
            np.testing.assert_allclose(candidates[2], expected_2)

    def test_nmax_scaling(self):
        """Test that nmax scales with acquisition_fun_return_size."""
        self.opt.acquisition_fun_return_size = 50
        self.opt.X_ = np.random.uniform(0, 1, size=(10, 2))

        with patch("spotoptim.SpotOptim.tricands") as mock_tricands:
            mock_tricands.return_value = np.zeros((100, 2))
            self.opt._acquisition_function = MagicMock(return_value=np.zeros(100))

            self.opt.optimize_acquisition_func()

            # nmax should be max(100*2, 50*50) = 2500
            _, kwargs = mock_tricands.call_args
            assert kwargs["nmax"] >= 2500
