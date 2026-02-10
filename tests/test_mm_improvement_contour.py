# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import matplotlib.pyplot as plt
import pytest
import importlib

# Import the submodule via importlib to avoid any name masking in package __init__
mm_module = importlib.import_module("spotoptim.sampling.mm")
from spotoptim.sampling.mm import mm_improvement_contour


def test_mm_improvement_contour_small_grid(monkeypatch):
    # Stub mm_improvement to a simple deterministic function: sum of coords
    def fake_mm_improvement(x, X_base, J_base, d_base, q=2, p=2, normalize_flag=True):
        return float(x[0] + x[1])

    monkeypatch.setattr(mm_module, "mm_improvement", fake_mm_improvement)

    captured = {}

    def fake_contourf(X1, X2, Z, **kwargs):
        captured["X1"] = X1
        captured["X2"] = X2
        captured["Z"] = Z
        captured["kwargs"] = kwargs
        class _Dummy: ...
        return _Dummy()

    monkeypatch.setattr(plt, "contourf", fake_contourf)
    monkeypatch.setattr(plt, "colorbar", lambda *args, **kw: None)
    monkeypatch.setattr(plt, "show", lambda *args, **kw: None)

    X_base = np.array([[0.1, 0.1], [0.2, 0.2], [0.7, 0.7]])
    x1 = np.array([0.0, 1.0])
    x2 = np.array([0.0, 1.0])

    mm_improvement_contour(X_base, x1=x1, x2=x2)

    assert "Z" in captured and "kwargs" in captured
    Z = captured["Z"]
    assert Z.shape == (2, 2)
    # With meshgrid default, Z[i, j] = x1[j] + x2[i]
    expected = np.array([[0.0, 1.0], [1.0, 2.0]])
    np.testing.assert_allclose(Z, expected, rtol=0, atol=0)
    assert captured["kwargs"].get("levels") == 30
    assert captured["kwargs"].get("cmap") == "viridis"


def test_mm_improvement_contour_default_grid_shapes(monkeypatch):
    # Keep it fast by returning a constant improvement
    monkeypatch.setattr(mm_module, "mm_improvement", lambda *args, **kw: 1.0)

    shape_holder = {}

    def fake_contourf(X1, X2, Z, **kwargs):
        shape_holder["Z_shape"] = Z.shape
        class _Dummy: ...
        return _Dummy()

    monkeypatch.setattr(plt, "contourf", fake_contourf)
    monkeypatch.setattr(plt, "colorbar", lambda *args, **kw: None)
    monkeypatch.setattr(plt, "show", lambda *args, **kw: None)

    X_base = np.array([[0.1, 0.1], [0.2, 0.2], [0.7, 0.7]])

    # Use defaults: np.linspace(0,1,100) for both x1 and x2 → 100x100 grid
    mm_improvement_contour(X_base)

    assert shape_holder.get("Z_shape") == (100, 100)


def test_mm_improvement_contour_scatter_called_with_base(monkeypatch):
    monkeypatch.setattr(mm_module, "mm_improvement", lambda *args, **kw: 1.0)

    scatter_calls = {}

    def fake_scatter(x, y, **kwargs):
        scatter_calls["x"] = np.asarray(x)
        scatter_calls["y"] = np.asarray(y)
        class _Dummy: ...
        return _Dummy()

    # Keep other plotting calls lightweight
    monkeypatch.setattr(plt, "scatter", fake_scatter)
    monkeypatch.setattr(plt, "contourf", lambda *a, **k: None)
    monkeypatch.setattr(plt, "colorbar", lambda *args, **kw: None)
    monkeypatch.setattr(plt, "show", lambda *args, **kw: None)

    X_base = np.array([[0.1, 0.1], [0.2, 0.2], [0.7, 0.7]])

    mm_improvement_contour(X_base)

    assert "x" in scatter_calls and "y" in scatter_calls
    np.testing.assert_allclose(scatter_calls["x"], X_base[:, 0])
    np.testing.assert_allclose(scatter_calls["y"], X_base[:, 1])
