import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


def _get_target_column(y, i):
    """Helper to extract the i-th column from y (DataFrame or numpy array)."""
    if hasattr(y, "iloc"):
        return y.iloc[:, i].values
    else:
        return y[:, i]


def mo_eval_models(
    X_train, y_train, X_test, y_test, model_define_func, scores=None, verbose=False
):
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
        verbose (bool): Whether to print verbose output.

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
        scoring_funcs = {"R2": r2_score}
        return_dict = False
    elif callable(scores):
        scoring_funcs = {"Score": scores}
        return_dict = False
    elif isinstance(scores, dict):
        scoring_funcs = scores
        return_dict = True
    else:
        raise ValueError(
            "scores argument must be None, a callable, or a dictionary of callables."
        )

    for i in range(target_count):
        # Fit pipeline for this target
        if verbose:
            print(f"Training model for target {i+1}/{target_count}...")
        model_pipeline = model_define_func()

        y_train_target = _get_target_column(y_train, i)

        model_pipeline.fit(X_train, y_train_target)
        models.append(model_pipeline)
        pred = model_pipeline.predict(X_test)
        preds.append(pred)

    # stack predictions for multi-output compatibility
    preds_stacked = np.column_stack(preds)

    # Calculate scores
    results = {name: [] for name in scoring_funcs}

    for i in range(target_count):
        y_true = _get_target_column(y_test, i)
        y_pred = preds_stacked[:, i]

        for name, func in scoring_funcs.items():
            results[name].append(func(y_true, y_pred))

    if not return_dict:
        # Return list of values for the single metric (backward compatibility)
        final_scores = list(results.values())[0]
        score_display = [f"{s:.2f}" for s in final_scores]
        if verbose:
            print(f"Model scores: {score_display}")
    else:
        final_scores = results
        if verbose:
            print("Model scores:")
            for name, vals in final_scores.items():
                score_display = [f"{s:.2f}" for s in vals]
                print(f"  {name}: {score_display}")

    if verbose:
        print(f"Predictions shape: {preds_stacked.shape}")
    return final_scores, models, preds_stacked


def mo_cv_models(X, y, model_define_func, cv=5, scores=None):
    """
    Performs cross-validation for separate models for each target in a multi-output problem.

    Args:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.DataFrame or np.ndarray): Target matrix with multiple columns.
        model_define_func (Callable): Function that returns a fresh model or pipeline instance.
        cv (int or cross-validation generator, default=5): Number of folds or CV object.
        scores (Union[Dict[str, str/Callable], str, Callable, None]): Scoring metric(s).
            - If None (default): Uses default scorer of the estimator. Returns List[np.ndarray].
            - If str/Callable: Single scorer. Returns List[np.ndarray].
            - If Dict: Dictionary of {name: scorer}. Returns Dict[str, List[np.ndarray]].
              Note: Unlike mo_eval_models which takes raw functions, cross_val_score expects
              strings (e.g. 'neg_mean_squared_error') or make_scorer callables, or raw callables that fit the sklearn signature.
              To maintain similarity with mo_eval_models, we rely on cross_val_score's flexibility.

    Returns:
        scores (Union[List[np.ndarray], Dict[str, List[np.ndarray]]]):
            - List of arrays (one array of CV scores per target) if `scores` is None or single.
            - Dictionary of {name: List[np.ndarray]} if `scores` is a dictionary.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from spotoptim.utils.eval import mo_cv_models

        >>> # Generate dummy data
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.rand(100, 5), columns=[f'x{i}' for i in range(5)])
        >>> y = pd.DataFrame(np.random.rand(100, 3), columns=[f'y{i}' for i in range(3)])

        >>> # Example 1: Default behavior (Default scorer, e.g. R2)
        >>> def make_model():
        ...     from sklearn.linear_model import Ridge
        ...     return Ridge()
        >>> cv_scores = mo_cv_models(X, y, make_model, cv=3)
        Cross-validating target 1/3...
        Cross-validating target 2/3...
        Cross-validating target 3/3...
        CV Scores Mean: ['-0.14', '-0.07', '-0.20']

        >>> # Example 2: Custom single score (NMSE - Negative Mean Squared Error)
        >>> # Note: sklearn scoring strings are preferred for cross_val_score
        >>> nmse_scores = mo_cv_models(X, y, make_model, cv=3, scores='neg_mean_squared_error')
        Cross-validating target 1/3...
        Cross-validating target 2/3...
        Cross-validating target 3/3...
        CV Scores Mean: ['-0.09', '-0.08', '-0.10']

        >>> # Example 3: Multiple custom scores
        >>> my_scores = {'R2': 'r2', 'NMSE': 'neg_mean_squared_error'}
        >>> all_cv_scores = mo_cv_models(X, y, make_model, cv=3, scores=my_scores)
        Cross-validating target 1/3...
        Cross-validating target 2/3...
        Cross-validating target 3/3...
        CV Scores Mean:
          R2: ['-0.14', '-0.07', '-0.20']
          NMSE: ['-0.09', '-0.08', '-0.10']
    """
    target_count = y.shape[1]

    # Determine scoring mode
    if scores is None:
        scoring_items = {"Score": None}  # None uses default estimator scorer
        return_dict = False
    elif isinstance(scores, (str, object)) and not isinstance(scores, dict):
        # 'object' to cover callable, but check not dict first
        scoring_items = {"Score": scores}
        return_dict = False
    elif isinstance(scores, dict):
        scoring_items = scores
        return_dict = True
    else:
        raise ValueError(
            "scores argument must be None, a string/callable, or a dictionary."
        )

    results = {name: [] for name in scoring_items}

    for i in range(target_count):
        print(f"Cross-validating target {i+1}/{target_count}...")
        y_target = _get_target_column(y, i)

        # Fresh model for each target (and cross_val_score clones it anyway)
        model = model_define_func()

        for name, scorer in scoring_items.items():
            # cross_val_score(estimator, X, y, scoring=..., cv=...)
            cv_res = cross_val_score(model, X, y_target, cv=cv, scoring=scorer)
            results[name].append(cv_res)

    if not return_dict:
        final_scores = list(results.values())[0]
        # Calculate means for display
        means = [f"{np.mean(s):.2f}" for s in final_scores]
        print(f"CV Scores Mean: {means}")
    else:
        final_scores = results
        print("CV Scores Mean:")
        for name, vals in final_scores.items():
            means = [f"{np.mean(s):.2f}" for s in vals]
            print(f"  {name}: {means}")

    return final_scores
