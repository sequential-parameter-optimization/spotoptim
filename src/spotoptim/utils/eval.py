import numpy as np
from sklearn.metrics import r2_score


def mo_eval_models(X_train, y_train, X_test, y_test, model_define_func, scores=None):
    """
    Trains and evaluates separate models for each target in a multi-output regression problem.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training feature matrix.
        y_train (pd.DataFrame or np.ndarray): Training target matrix with multiple columns (one per target).
        X_test (pd.DataFrame or np.ndarray): Test feature matrix.
        y_test (pd.DataFrame or np.ndarray): Test target matrix with multiple columns (one per target).
        model_define_func (Callable): Function that returns a fresh model or pipeline instance for training.
        scores (Union[Dict[str, Callable], Callable, None]): Scoring metric(s) to evaluate.
            - If None (default): Uses R2 score. Returns List[float].
            - If Callable: Single scoring function (e.g., mean_squared_error). Returns List[float].
            - If Dict: Dictionary of {name: scoring_func}. Returns Dict[str, List[float]].

    Returns:
        scores (Union[List[float], Dict[str, List[float]]]): 
            - List of scores if `scores` is None or a single callable.
            - Dictionary of scores if `scores` is a dictionary.
        models (List): List of trained model instances, one per target.
        preds_stacked (np.ndarray): Array of stacked predictions for all targets, shape (n_samples, n_targets).

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from spotoptim.utils.eval import mo_eval_models
        
        >>> # Generate dummy data
        >>> np.random.seed(42)
        >>> X_train = pd.DataFrame(np.random.rand(100, 5), columns=[f'x{i}' for i in range(5)])
        >>> y_train = pd.DataFrame(np.random.rand(100, 3), columns=[f'y{i}' for i in range(3)])
        >>> X_test = pd.DataFrame(np.random.rand(20, 5), columns=[f'x{i}' for i in range(5)])
        >>> y_test = pd.DataFrame(np.random.rand(20, 3), columns=[f'y{i}' for i in range(3)])

        >>> # Example 1: Default behavior (R2 score)
        >>> def make_model():
        ...     from sklearn.linear_model import Ridge
        ...     return Ridge()
        >>> r2_scores, models, preds = mo_eval_models(X_train, y_train, X_test, y_test, make_model)
        Training model for target 1/3...
        Training model for target 2/3...
        Training model for target 3/3...
        Model scores: ['-0.10', '-0.13', '-0.19']
        Predictions shape: (20, 3)

        >>> # Example 2: Custom single score (MSE)
        >>> from sklearn.metrics import mean_squared_error
        >>> mse_scores, _, _ = mo_eval_models(X_train, y_train, X_test, y_test, make_model, scores=mean_squared_error)
        Training model for target 1/3...
        Training model for target 2/3...
        Training model for target 3/3...
        Model scores: ['0.07', '0.09', '0.10']
        Predictions shape: (20, 3)

        >>> # Example 3: Multiple custom scores
        >>> from sklearn.metrics import mean_absolute_error, r2_score
        >>> my_scores = {'R2': r2_score, 'MSE': mean_squared_error, 'MAE': mean_absolute_error}
        >>> all_scores, _, _ = mo_eval_models(X_train, y_train, X_test, y_test, make_model, scores=my_scores)
        Training model for target 1/3...
        Training model for target 2/3...
        Training model for target 3/3...
        Model scores:
          R2: ['-0.10', '-0.13', '-0.19']
          MSE: ['0.07', '0.09', '0.10']
          MAE: ['0.21', '0.27', '0.28']
        Predictions shape: (20, 3)
    """
    models = []
    preds = []
    target_count = y_train.shape[1]
    
    # Determine scoring mode
    if scores is None:
        scoring_funcs = {'R2': r2_score}
        return_dict = False
    elif callable(scores):
        scoring_funcs = {'Score': scores}
        return_dict = False
    elif isinstance(scores, dict):
        scoring_funcs = scores
        return_dict = True
    else:
        raise ValueError("scores argument must be None, a callable, or a dictionary of callables.")

    for i in range(target_count):
        # Fit pipeline for this target
        print(f"Training model for target {i+1}/{target_count}...")
        model_pipeline = model_define_func()
        # Handle pandas vs numpy indexing
        if hasattr(y_train, 'iloc'):
            y_train_target = y_train.iloc[:, i].values
        else:
            y_train_target = y_train[:, i]

        model_pipeline.fit(X_train, y_train_target)
        models.append(model_pipeline)
        pred = model_pipeline.predict(X_test)
        preds.append(pred)

    # stack predictions for multi-output compatibility
    preds_stacked = np.column_stack(preds)

    # Calculate scores
    results = {name: [] for name in scoring_funcs}
    
    for i in range(target_count):
        if hasattr(y_test, 'iloc'):
            y_true = y_test.iloc[:, i].values
        else:
            y_true = y_test[:, i]
        y_pred = preds_stacked[:, i]
        
        for name, func in scoring_funcs.items():
            results[name].append(func(y_true, y_pred))

    if not return_dict:
        # Return list of values for the single metric (backward compatibility)
        final_scores = list(results.values())[0]
        score_display = [f'{s:.2f}' for s in final_scores]
        print(f"Model scores: {score_display}")
    else:
        final_scores = results
        print("Model scores:")
        for name, vals in final_scores.items():
            score_display = [f'{s:.2f}' for s in vals]
            print(f"  {name}: {score_display}")

    print(f"Predictions shape: {preds_stacked.shape}")
    return final_scores, models, preds_stacked
