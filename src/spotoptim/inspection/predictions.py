import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay


def plot_actual_vs_predicted(
    y_test, y_pred, title=None, show=True, filename=None
) -> None:
    """Plot actual vs. predicted values.

    Args:
        y_test (np.ndarray):
            True values.
        y_pred (np.ndarray):
            Predicted values.
        title (str, optional):
            Title of the plot. Defaults to None.
        show (bool, optional):
            If True, the plot is shown. Defaults to True.
        filename (str, optional):
            Name of the file to save the plot. Defaults to None.

    Returns:
        (NoneType): None

    Examples:
        >>> from sklearn.datasets import load_diabetes
            from sklearn.linear_model import LinearRegression
            from spotoptim.inspection import plot_actual_vs_predicted
            X, y = load_diabetes(return_X_y=True)
            lr = LinearRegression()
            lr.fit(X, y)
            y_pred = lr.predict(X)
            plot_actual_vs_predicted(y, y_pred)
    """
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    PredictionErrorDisplay.from_predictions(
        y_test,
        y_pred=y_pred,
        kind="actual_vs_predicted",
        subsample=100,
        ax=axs[0],
        random_state=0,
        scatter_kwargs={"alpha": 0.5},
    )
    axs[0].set_title("Actual vs. Predicted values")
    PredictionErrorDisplay.from_predictions(
        y_test,
        y_pred=y_pred,
        kind="residual_vs_predicted",
        subsample=100,
        ax=axs[1],
        random_state=0,
    )
    axs[1].set_title("Residuals vs. Predicted Values")
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()
