# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from sklearn.decomposition import PCA
from spotoptim.utils.pca import (
    get_pca,
    plot_pca_scree,
    plot_pca1vs2,
    get_pca_topk,
    get_loading_scores,
    plot_loading_scores,
)


@pytest.fixture
def sample_data():
    """Create a simple dataframe for testing."""
    df = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [2.0, 4.0, 6.0, 8.0, 10.0],
            "C": [5.0, 4.0, 3.0, 2.0, 1.0],
            "Label": ["X", "Y", "Z", "W", "V"],  # Non-numeric
        }
    )
    return df


@pytest.fixture
def fitted_pca(sample_data):
    """Return a fitted PCA object and related data."""
    numeric_df = sample_data.select_dtypes(include=[np.number])
    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(numeric_df)
    return pca, pca_data, numeric_df.columns


def test_get_pca(sample_data):
    """Test get_pca function."""
    pca, pca_scores, feature_names, sample_names, df_pca_components = get_pca(
        sample_data, n_components=2
    )

    assert isinstance(pca, PCA)
    assert pca.n_components == 2
    assert pca_scores.shape == (5, 2)
    assert len(feature_names) == 3
    assert "Label" not in feature_names
    assert len(sample_names) == 5
    assert isinstance(df_pca_components, pd.DataFrame)
    assert list(df_pca_components.columns) == ["PC1", "PC2"]


@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.figure")
def test_plot_pca_scree(mock_figure, mock_show, fitted_pca):
    """Test plot_pca_scree function."""
    pca, _, _ = fitted_pca

    # Test with default args
    plot_pca_scree(pca)
    mock_show.assert_called()

    # Test with max_scree
    plot_pca_scree(pca, max_scree=2, df_name="Test DF")
    mock_show.assert_called()


@patch("matplotlib.pyplot.scatter")
@patch("matplotlib.pyplot.figure")
def test_plot_pca1vs2(mock_figure, mock_scatter, fitted_pca):
    """Test plot_pca1vs2 function."""
    pca, pca_data, _ = fitted_pca

    plot_pca1vs2(pca, pca_data, df_name="Test DF")
    mock_scatter.assert_called()
    mock_figure.assert_called()


def test_get_pca_topk(fitted_pca):
    """Test get_pca_topk function."""
    pca, _, feature_names = fitted_pca

    # Test with k=1
    top_pc1, top_pc2 = get_pca_topk(pca, feature_names, k=1)

    assert isinstance(top_pc1, list)
    assert isinstance(top_pc2, list)
    assert len(top_pc1) == 1
    assert len(top_pc2) == 1
    assert top_pc1[0] in feature_names

    # Test with k=3 (all features)
    top_pc1, top_pc2 = get_pca_topk(pca, feature_names, k=3)
    assert len(top_pc1) == 3


def test_get_loading_scores(fitted_pca):
    """Test get_loading_scores function."""
    pca, _, feature_names = fitted_pca

    scores = get_loading_scores(pca, feature_names)

    assert isinstance(scores, pd.DataFrame)
    assert scores.shape == (3, 3)  # 3 features (rows) x 3 PCs (cols)
    assert list(scores.index) == list(feature_names)
    assert "PC1" in scores.columns


@patch("matplotlib.pyplot.show")
@patch("seaborn.heatmap")
@patch("matplotlib.pyplot.figure")
def test_plot_loading_scores(mock_figure, mock_heatmap, mock_show, fitted_pca):
    """Test plot_loading_scores function."""
    pca, _, feature_names = fitted_pca
    scores = get_loading_scores(pca, feature_names)

    plot_loading_scores(scores)

    mock_heatmap.assert_called()
    mock_show.assert_called()
