# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for plot_mmphi_vs_n_lhs in spotoptim.sampling.mm
"""

from unittest.mock import patch
from spotoptim.sampling.mm import plot_mmphi_vs_n_lhs


class TestPlotMmphiVsNLhsBasic:
    @patch("matplotlib.pyplot.show")
    def test_basic_execution_calls_show(self, mock_show):
        plot_mmphi_vs_n_lhs(
            k_dim=3, seed=42, n_min=5, n_max=10, n_step=5, q_phi=2.0, p_phi=2.0
        )
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_empty_range_prints_warning_and_no_show(self, mock_show, capsys):
        plot_mmphi_vs_n_lhs(k_dim=3, seed=42, n_min=10, n_max=5, n_step=5)
        captured = capsys.readouterr()
        assert "Warning: n_values list is empty" in captured.out
        mock_show.assert_not_called()


class TestPlotMmphiVsNLhsEdgeCases:
    @patch("spotoptim.sampling.mm.mmphi")
    @patch("matplotlib.pyplot.show")
    def test_exception_in_mmphi_handled(self, mock_show, mock_mmphi, capsys):
        # First call raises, subsequent calls return a float
        def side_effect(*args, **kwargs):
            if not hasattr(side_effect, "called"):
                side_effect.called = True
                raise ValueError("boom")
            return 1.23

        mock_mmphi.side_effect = side_effect
        plot_mmphi_vs_n_lhs(k_dim=2, seed=1, n_min=2, n_max=4, n_step=1)
        out = capsys.readouterr().out
        assert "Error calculating for n=2" in out
        mock_show.assert_called_once()

    @patch("spotoptim.sampling.mm.mmphi_intensive")
    @patch("matplotlib.pyplot.show")
    def test_mmphi_intensive_called_and_handled(self, mock_show, mock_mmphi_intensive):
        # Return tuple as expected (phi_intensive, J, d)
        mock_mmphi_intensive.return_value = (0.5, None, None)
        plot_mmphi_vs_n_lhs(k_dim=2, seed=1, n_min=2, n_max=2, n_step=1)
        mock_mmphi_intensive.assert_called()
        mock_show.assert_called_once()
