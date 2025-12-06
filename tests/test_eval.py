
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
