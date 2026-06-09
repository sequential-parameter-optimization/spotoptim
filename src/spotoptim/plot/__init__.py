# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plotting and visualization utilities for SpotOptim.

Plotting helpers live in submodules (``contour``, ``mo``, ``visualization``)
and import matplotlib lazily, so importing this package does not pull in
matplotlib. Import the helper you need directly from its submodule, e.g.
``from spotoptim.plot.visualization import plot_progress``.
"""
