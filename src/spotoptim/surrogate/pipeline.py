from typing import Any


class Pipeline:
    """
    Pipeline of transforms with a final estimator.

    Sequentially apply a list of transforms and a final estimator.
    This simple implementation assumes that all steps except the last one
    are transformers (have `fit_transform` or `fit`+`transform`), and the last
    step is an estimator (has `fit` and `predict`).

    Args:
        steps (list): List of (name, transform) tuples (implementing fit/transform)
            that are chained, in the order they are chained, with the last object
            an estimator.
    """

    def __init__(self, steps):
        self.steps = steps

    @property
    def _final_estimator(self):
        """Returns the final estimator of the pipeline."""
        return self.steps[-1][1]

    def fit(self, X, y=None, **fit_params) -> "Pipeline":
        """
        Fit the model.

        Fit all the transformers one after the other and transform the
        data. Finally, fit the transformed data using the final estimator.

        Args:
            X (iterable): Training data. Must fulfill input requirements of first step of the pipeline.
            y (iterable, optional): Training targets. Must fulfill label requirements for all steps of the pipeline.
            **fit_params (Any): Additional parameters passed to the `fit` method of the final estimator.

        Returns:
            Pipeline: Pipeline with fitted steps.
        """
        Xt = X
        for name, transform in self.steps[:-1]:
            if hasattr(transform, "fit_transform"):
                Xt = transform.fit_transform(Xt, y)
            else:
                Xt = transform.fit(Xt, y).transform(Xt)

        if hasattr(self._final_estimator, "fit"):
            self._final_estimator.fit(Xt, y, **fit_params)

        return self

    def predict(self, X, **predict_params) -> Any:
        """
        Transform the data, and apply `predict` with the final estimator.

        Args:
            X (iterable): Data to predict on. Must fulfill input requirements of first step of the pipeline.
            **predict_params (Any): Additional parameters passed to the `predict` method of the final estimator.

        Returns:
            Any: Result of calling `predict` on the final estimator.
        """
        Xt = X
        for name, transform in self.steps[:-1]:
            if hasattr(transform, "transform"):
                Xt = transform.transform(Xt)

        return self._final_estimator.predict(Xt, **predict_params)
