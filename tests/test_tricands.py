# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for tricands functionality in spotoptim.
"""

import pytest
import numpy as np
from unittest.mock import patch
from spotoptim.tricands import tricands
from spotoptim.tricands.tricands import tricands_interior, tricands_fringe


class TestTricands:
    """Test suite for tricands module."""

    def setup_method(self):
        # Create invalid (too few points) X
        self.X_small = np.array([[0.1, 0.1]])
        # Create valid 2D X
        self.X_2d = np.array(
            [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9], [0.2, 0.5], [0.8, 0.8], [0.4, 0.4]]
        )
        # Create valid 3D X
        self.X_3d = np.random.rand(10, 3)

    def test_tricands_interior(self):
        """Test tricands_interior subroutine."""
        res = tricands_interior(self.X_2d)
        assert "cand" in res
        assert "tri" in res
        assert isinstance(res["cand"], np.ndarray)
        assert res["cand"].shape[1] == 2

        # Test exception for too few points
        with pytest.raises(Exception, match=r"must have nrow\(X\) >= ncol\(X\) \+ 1"):
            tricands_interior(self.X_small)

    def test_tricands_fringe(self):
        """Test tricands_fringe subroutine."""
        res = tricands_fringe(self.X_2d, p=0.5)
        assert "XF" in res
        assert "XB" in res
        assert "qhull" in res
        assert isinstance(res["XF"], np.ndarray)
        assert res["XF"].shape[1] == 2

        # Test with p=0 (on hull)
        _ = tricands_fringe(self.X_2d, p=0.0)
        # Should be closer to hull than p=0.5/bound, tough to specific assert without geometry check
        # But we ensure it runs

        # Test exception for too few points
        with pytest.raises(Exception, match=r"must have nrow\(X\) >= ncol\(X\) \+ 1"):
            tricands_fringe(self.X_small)

    def test_tricands_basic(self):
        """Test basic tricands execution."""
        candidates = tricands(self.X_2d, fringe=False)
        assert isinstance(candidates, np.ndarray)
        assert candidates.shape[1] == 2

        # With fringe
        candidates_fringe = tricands(self.X_2d, fringe=True)
        assert candidates_fringe.shape[0] > candidates.shape[0]

    def test_tricands_high_dim(self):
        """Test tricands on >2D data."""
        candidates = tricands(self.X_3d, fringe=True)
        assert candidates.shape[1] == 3

    def test_bounds_check(self):
        """Test input bounds checking."""
        X_out = np.array([[1.5, 0.5], [0.5, 0.5], [0.1, 0.1]])
        with pytest.raises(Exception, match="X outside of lower/upper bounds"):
            tricands(X_out, lower=0, upper=1)

    def test_visualization_exception(self):
        """Test visual exception for dim != 2."""
        with pytest.raises(
            Exception, match=r"visuals only possible when ncol\(X\) = 2"
        ):
            tricands(self.X_3d, vis=True)

    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_visualization_2d(self, mock_close, mock_savefig, mock_figure):
        """Test that visualization code runs for 2D data."""
        tricands(self.X_2d, vis=True, fringe=True)
        mock_figure.assert_called()
        mock_savefig.assert_called()
        mock_close.assert_called()

    def test_subsetting_best(self):
        """Test subsetting with 'best' argument."""
        # Force nmax small to trigger subsetting
        X_large = np.random.rand(50, 2)
        candidates = tricands(X_large, fringe=True, nmax=10, best=0)
        assert candidates.shape[0] == 10

    def test_subsetting_ordering(self):
        """Test subsetting with 'ordering' argument."""
        X_large = np.random.rand(50, 2)
        ordering = np.arange(50)  # Mock ordering
        candidates = tricands(X_large, fringe=True, nmax=10, ordering=ordering)
        assert candidates.shape[0] == 10

    def test_subsetting_conflict(self):
        """Test exception when both best and ordering are provided."""
        with pytest.raises(Exception, match="can only subset for BO or CL, not both"):
            tricands(self.X_2d, best=0, ordering=np.arange(6))
