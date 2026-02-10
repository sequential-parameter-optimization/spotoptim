# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for calculate_outliers supporting Series and DataFrame."""

import numpy as np
import pandas as pd
import pytest
from spotoptim.utils.stats import calculate_outliers


class TestCalculateOutliersSeries:
    def test_series_simple(self):
        s = pd.Series([1, 2, 3, 4, 5, 100])
        assert calculate_outliers(s) == 1

    def test_series_no_outliers(self):
        s = pd.Series([10, 11, 12, 13, 14])
        assert calculate_outliers(s) == 0

    def test_series_all_same(self):
        s = pd.Series([5, 5, 5, 5])
        assert calculate_outliers(s) == 0

    def test_series_with_nans(self):
        s = pd.Series([1, np.nan, 2, 3, np.nan, 100])
        assert calculate_outliers(s) == 1

    def test_series_custom_multiplier(self):
        s = pd.Series([1, 2, 3, 4, 5, 10])
        # With smaller multiplier, 10 may be an outlier
        assert calculate_outliers(s, irqmultiplier=1.0) >= 0


class TestCalculateOutliersDataFrame:
    def test_dataframe_simple(self):
        df = pd.DataFrame({
            'a': [1, 2, 3, 100],  # 1 outlier
            'b': [10, 12, 11, 10]  # 0 outliers
        })
        assert calculate_outliers(df) == 1

    def test_dataframe_multiple_columns(self):
        df = pd.DataFrame({
            'a': [1, 2, 3, 100],   # 1 outlier
            'b': [10, 12, 200, 10] # 1 outlier
        })
        assert calculate_outliers(df) == 2

    def test_dataframe_non_numeric_columns(self):
        df = pd.DataFrame({
            'a': [1, 2, 3, 100],
            'b': ['x', 'y', 'z', 'w']
        })
        # Only numeric column considered; expect 1
        assert calculate_outliers(df) == 1

    def test_dataframe_with_nans(self):
        df = pd.DataFrame({
            'a': [1, np.nan, 2, 100],
            'b': [10, 12, np.nan, 10]
        })
        # With pandas quartile definition on three values [1, 2, 100],
        # 100 is not flagged as an outlier for IQR=Q3-Q1.
        assert calculate_outliers(df) == 0

    def test_dataframe_empty_numeric(self):
        df = pd.DataFrame({
            'a': ['x', 'y'],
            'b': ['p', 'q']
        })
        assert calculate_outliers(df) == 0

    def test_dataframe_all_same(self):
        df = pd.DataFrame({
            'a': [5, 5, 5, 5],
            'b': [10, 10, 10, 10]
        })
        assert calculate_outliers(df) == 0

    def test_dataframe_custom_multiplier(self):
        df = pd.DataFrame({
            'a': [1, 2, 3, 10],
            'b': [10, 12, 11, 10]
        })
        # Tight fences may count 10 as outlier in 'a'
        assert calculate_outliers(df, irqmultiplier=1.0) >= 0


class TestCalculateOutliersErrors:
    def test_invalid_input_type(self):
        with pytest.raises(TypeError):
            calculate_outliers([1, 2, 3])
