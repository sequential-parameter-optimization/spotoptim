# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
from spotoptim.factor_analyzer import (
    ConfirmatoryFactorAnalyzer,
    ModelSpecificationParser,
)
from spotoptim.utils import get_internal_datasets_folder
import os


def test_cfa_main_example():
    """Test the main example in ConfirmatoryFactorAnalyzer docstring."""
    X = pd.read_csv(os.path.join(get_internal_datasets_folder(), "test11.csv"))
    model_dict = {"F1": ["V1", "V2", "V3", "V4"], "F2": ["V5", "V6", "V7", "V8"]}
    model_spec = ModelSpecificationParser.parse_model_specification_from_dict(
        X, model_dict
    )
    cfa = ConfirmatoryFactorAnalyzer(model_spec, disp=False)
    cfa = cfa.fit(X.values)

    # Check loadings
    loadings_expected = np.array(
        [
            [0.99, 0.0],
            [0.46, 0.0],
            [0.35, 0.0],
            [0.58, 0.0],
            [0.0, 0.99],
            [0.0, 0.73],
            [0.0, 0.38],
            [0.0, 0.5],
        ]
    )
    np.testing.assert_array_almost_equal(np.round(cfa.loadings_, 2), loadings_expected)

    # Check factor varcovs
    varcovs_expected = np.array([[1.0, 0.17], [0.17, 1.0]])
    np.testing.assert_array_almost_equal(
        np.round(cfa.factor_varcovs_, 2), varcovs_expected
    )

    # Check standard errors
    loadings_se, variances_se = cfa.get_standard_errors()
    loadings_se_expected = np.array(
        [
            [0.07, 0.0],
            [0.04, 0.0],
            [0.04, 0.0],
            [0.05, 0.0],
            [0.0, 0.06],
            [0.0, 0.05],
            [0.0, 0.04],
            [0.0, 0.04],
        ]
    )
    np.testing.assert_array_almost_equal(np.round(loadings_se, 2), loadings_se_expected)

    variances_se_expected = np.array([0.12, 0.05, 0.05, 0.06, 0.1, 0.07, 0.05, 0.05])
    np.testing.assert_array_almost_equal(
        np.round(variances_se, 2), variances_se_expected
    )

    # Check transform
    scores = cfa.transform(X.values)
    scores_subset_expected = np.array([[-0.47, -1.09], [2.59, 1.2], [-0.47, 2.66]])
    np.testing.assert_array_almost_equal(
        np.round(scores[:3], 2), scores_subset_expected
    )


def test_cfa_get_model_implied_cov():
    """Test the get_model_implied_cov example."""
    X = pd.read_csv(os.path.join(get_internal_datasets_folder(), "test11.csv"))
    model_dict = {"F1": ["V1", "V2", "V3", "V4"], "F2": ["V5", "V6", "V7", "V8"]}
    model_spec = ModelSpecificationParser.parse_model_specification_from_dict(
        X, model_dict
    )
    cfa = ConfirmatoryFactorAnalyzer(model_spec, disp=False)
    cfa = cfa.fit(X.values)

    implied_cov = cfa.get_model_implied_cov()
    expected_cov = np.array(
        [
            [2.08, 0.46, 0.35, 0.58, 0.17, 0.13, 0.06, 0.09],
            [0.46, 1.17, 0.16, 0.27, 0.08, 0.06, 0.03, 0.04],
            [0.35, 0.16, 1.07, 0.2, 0.06, 0.04, 0.02, 0.03],
            [0.58, 0.27, 0.2, 1.29, 0.1, 0.07, 0.04, 0.05],
            [0.17, 0.08, 0.06, 0.1, 2.04, 0.72, 0.37, 0.49],
            [0.13, 0.06, 0.04, 0.07, 0.72, 1.48, 0.28, 0.37],
            [0.06, 0.03, 0.02, 0.04, 0.37, 0.28, 1.12, 0.19],
            [0.09, 0.04, 0.03, 0.05, 0.49, 0.37, 0.19, 1.29],
        ]
    )
    np.testing.assert_array_almost_equal(np.round(implied_cov, 2), expected_cov)
