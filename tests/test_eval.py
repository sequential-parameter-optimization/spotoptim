# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

from spotoptim.utils.eval import mo_eval_models


@pytest.fixture
def data():
    np.random.seed(42)
    n_samples = 50
    n_features = 5
    n_targets = 3
    
    X = np.random.rand(n_samples, n_features)
    # y = X @ W + noise
    W = np.random.rand(n_features, n_targets)
    y = X @ W + 0.1 * np.random.randn(n_samples, n_targets)
    
    # Split
    split_idx = int(n_samples * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, y_train, X_test, y_test

def make_model():
    return Ridge()

def test_mo_eval_models_default_score(data):
    X_train, y_train, X_test, y_test = data
    # Pass as dataframe to test iloc handling too, although code handles numpy now too
    y_train_df = pd.DataFrame(y_train)
    y_test_df = pd.DataFrame(y_test)
    
    scores, models, preds = mo_eval_models(X_train, y_train_df, X_test, y_test_df, make_model)
    
    assert len(models) == 3
    assert preds.shape == (10, 3) # 10 test samples
    assert isinstance(scores, list)
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)
    # Should perform reasonably well
    assert all(s > 0.0 for s in scores)

def test_mo_eval_models_single_callable(data):
    X_train, y_train, X_test, y_test = data
    
    scores, models, preds = mo_eval_models(X_train, y_train, X_test, y_test, make_model, scores=mean_squared_error)
    
    assert isinstance(scores, list)
    assert len(scores) == 3
    # MSE should be positive
    assert all(s >= 0.0 for s in scores)

def test_mo_eval_models_dict_scores(data):
    X_train, y_train, X_test, y_test = data
    
    my_scores = {
        'R2': r2_score,
        'MSE': mean_squared_error
    }
    
    scores_dict, models, preds = mo_eval_models(X_train, y_train, X_test, y_test, make_model, scores=my_scores)
    
    assert isinstance(scores_dict, dict)
    assert 'R2' in scores_dict
    assert 'MSE' in scores_dict
    assert len(scores_dict['R2']) == 3
    assert len(scores_dict['MSE']) == 3

def test_mo_eval_models_numpy_input(data):
    X_train, y_train, X_test, y_test = data # These are numpy arrays
    
    scores, models, preds = mo_eval_models(X_train, y_train, X_test, y_test, make_model)
    
    assert len(scores) == 3

def test_mo_cv_models_basic(data):
    X_train, y_train, _, _ = data
    
    from spotoptim.utils.eval import mo_cv_models
    
    # Test default (R2 usually)
    cv_scores = mo_cv_models(X_train, y_train, make_model, cv=3)
    
    assert isinstance(cv_scores, list)
    assert len(cv_scores) == 3 # 3 targets
    assert len(cv_scores[0]) == 3 # 3 folds
    assert all(isinstance(s, np.ndarray) for s in cv_scores)

def test_mo_cv_models_single_score_str(data):
    X_train, y_train, _, _ = data
    from spotoptim.utils.eval import mo_cv_models
    
    # Test with string scorer
    cv_scores = mo_cv_models(X_train, y_train, make_model, cv=3, scores='neg_mean_squared_error')
    
    assert isinstance(cv_scores, list)
    assert len(cv_scores) == 3
    assert len(cv_scores[0]) == 3
    # NMSE should be negative
    assert all(s.mean() < 0 for s in cv_scores)

def test_mo_cv_models_dict_scores(data):
    X_train, y_train, _, _ = data
    from spotoptim.utils.eval import mo_cv_models
    
    my_scores = {'R2': 'r2', 'NMSE': 'neg_mean_squared_error'}
    cv_scores = mo_cv_models(X_train, y_train, make_model, cv=3, scores=my_scores)
    
    assert isinstance(cv_scores, dict)
    assert 'R2' in cv_scores
    assert 'NMSE' in cv_scores
    assert len(cv_scores['R2']) == 3
