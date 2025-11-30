import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from spotoptim.sampling.mm import mmphi_intensive, mmphi_intensive_update
from scipy.optimize import dual_annealing
from spotdesirability.utils.desirability import DMin, DMax, DOverall
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate artificial data
n_samples = 50
n_features = 2

# X: Random points in [0, 1]^2
X = np.random.rand(n_samples, n_features)

# y1: Sphere function (minimize)
y1 = np.sum((X - 0.5)**2, axis=1)

# y2: Rosenbrock-like function (minimize)
y2 = 100 * (X[:, 1] - X[:, 0]**2)**2 + (1 - X[:, 0])**2

# Train Random Forest models
model_y1 = RandomForestRegressor(n_estimators=100, random_state=42)
model_y1.fit(X, y1)

model_y2 = RandomForestRegressor(n_estimators=100, random_state=42)
model_y2.fit(X, y2)

print("Models trained successfully.")

# Define desirability functions using spotdesirability
# Target 1 (y1): Minimize
d1 = DMin(y1.min(), y1.max())

# Target 2 (y2): Minimize
d2 = DMin(y2.min(), y2.max())

# Target 3 (y_mm): Maximize space-filling improvement
# y_mm = Phi_base - Phi_new
# We estimate bounds for y_mm
X_base = X.copy()
phi_base, J_base, d_base = mmphi_intensive(X_base, q=2, p=2)

ymm_min = -0.5 # Allow for some degradation
ymm_max = 0.5  # Expected improvement
d_mm = DMax(ymm_min, ymm_max)

# Combine into overall desirability
# We use geometric mean (default for DOverall)
D_overall = DOverall(d1, d2, d_mm)

def multi_objective(x, models, X_base, J, d, phi_base):
    """
    Calculates the negative combined desirability for a candidate point x.
    
    Args:
        x (np.ndarray): Candidate point (1D array).
        models (list): List of trained models [model_y1, model_y2].
        X_base (np.ndarray): Existing design points.
        J (np.ndarray): Multiplicities of distances for X_base.
        d (np.ndarray): Unique distances for X_base.
        phi_base (float): Base Morris-Mitchell metric for X_base.
        
    Returns:
        float: Negative geometric mean of desirabilities (for minimization).
    """
    # 1. Predict y1 and y2
    # Reshape x for prediction (1, n_features)
    x_reshaped = x.reshape(1, -1)
    y1_pred = models[0].predict(x_reshaped)[0]
    y2_pred = models[1].predict(x_reshaped)[0]
    
    # 2. Compute y_mm (Space-filling improvement)
    # Use efficient update
    phi_new, _, _ = mmphi_intensive_update(X_base, x, J, d, q=2, p=2)
    y_mm = phi_base - phi_new
    
    # 3. Calculate combined desirability
    # We pass the values as a list to the overall desirability object
    D = D_overall.predict([y1_pred, y2_pred, y_mm])
    
    # Return negative D because optimizers usually minimize
    # Also return the individual values for logging
    return -D, [y1_pred, y2_pred, y_mm]

def man_optim(X_base, models, bounds):
    """
    Optimizes the multi-objective function to find the next best point.
    Returns the best point, its desirability, and the history of objective values.
    """
    # Pre-calculate base MM stats
    phi_base, J_base, d_base = mmphi_intensive(X_base, q=2, p=2)
    
    # List to store callback values
    callback_values = []
    
    # Define the objective wrapper
    def func(x):
        neg_D, objectives = multi_objective(x, models, X_base, J_base, d_base, phi_base)
        callback_values.append(objectives)
        return neg_D
    
    # Run optimization
    result = dual_annealing(func, bounds=bounds, maxiter=100, seed=42)
    
    return result.x, -result.fun, np.array(callback_values)

# Define bounds for the design variables (0 to 1)
bounds = [(0, 1), (0, 1)]

# Ensure X_base and MM stats are available (in case of partial execution)
if 'X_base' not in locals():
    X_base = X.copy()
    phi_base, J_base, d_base = mmphi_intensive(X_base, q=2, p=2)

# Find the next best point
best_x, best_desirability, callback_values = man_optim(X_base, [model_y1, model_y2], bounds)

print(f"Best new point: {best_x}")
print(f"Predicted Desirability: {best_desirability:.4f}")

# Verify predictions for the new point
y1_pred = model_y1.predict(best_x.reshape(1, -1))[0]
y2_pred = model_y2.predict(best_x.reshape(1, -1))[0]
phi_new, _, _ = mmphi_intensive_update(X_base, best_x, J_base, d_base, q=2, p=2)
y_mm = phi_base - phi_new

print(f"Predicted y1: {y1_pred:.4f}")
print(f"Predicted y2: {y2_pred:.4f}")
print(f"Space-filling improvement (y_mm): {y_mm:.4f}")

# Visualization using plot_mo from index.qmd
def is_pareto_efficient(costs: np.ndarray, minimize: bool = True) -> np.ndarray:
    """
    Find the Pareto-efficient points from a set of points.

    A point is Pareto-efficient if no other point exists that is better in all objectives.
    This function assumes that lower values are preferred for each objective when `minimize=True`,
    and higher values are preferred when `minimize=False`.

    Args:
        costs (np.ndarray):
            An (N,M) array-like object of points, where N is the number of points and M is the number of objectives.
        minimize (bool, optional):
            If True, the function finds Pareto-efficient points assuming
            lower values are better. If False, it assumes higher values are better.
            Defaults to True.

    Returns:
        np.ndarray:
            A boolean mask of length N, where True indicates that the corresponding point is Pareto-efficient.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, cost in enumerate(costs):
        if is_efficient[i]:
            if minimize:
                is_efficient[is_efficient] = np.any(costs[is_efficient] < cost, axis=1)
            else:
                is_efficient[is_efficient] = np.any(costs[is_efficient] > cost, axis=1)
            is_efficient[i] = True
    return is_efficient

def plot_mo(
    combinations: list,
    pareto: str,
    y_rf: np.ndarray = None,
    pareto_front_y_rf: bool = False,
    y_best: np.ndarray = None,
    y_orig: np.ndarray = None,
    pareto_front_orig: bool = False,
    pareto_label: bool = False,
    target_names: list = None,
    y_rf_color="blue",
    y_best_color="red",
    title: str = "",
):
    """
    Generates scatter plots for each combination of two targets from a multi-output prediction while highlighting Pareto optimal points.
    """
    # Convert y_rf to numpy array if it's a pandas DataFrame
    if isinstance(y_rf, pd.DataFrame):
        y_rf = y_rf.values

    # Convert y_orig to numpy array if it's a pandas DataFrame
    if isinstance(y_orig, pd.DataFrame):
        y_orig = y_orig.values

    for i, j in combinations:
        plt.figure(figsize=(8, 6))
        s = 50  # Base size for points
        pareto_size = s  # Size for Pareto points
        if pareto_label:
            pareto_size = s * 4  # Increase the size for Pareto points
        a = 0.4

        # Plot original data if provided
        if y_orig is not None:
            # Determine Pareto optimal points for original data
            minimize = pareto == "min"
            pareto_mask_orig = is_pareto_efficient(y_orig[:, [i, j]], minimize)

            # Plot all original points
            plt.scatter(y_orig[:, i], y_orig[:, j], edgecolor="w", c="gray", s=s, marker="o", alpha=a, label="Original Points")

            # Highlight Pareto points for original data
            plt.scatter(y_orig[pareto_mask_orig, i], y_orig[pareto_mask_orig, j], edgecolor="k", c="gray", s=pareto_size, marker="o", alpha=a, label="Original Pareto")

            # Label Pareto points for original data if requested
            if pareto_label:
                for idx in np.where(pareto_mask_orig)[0]:
                    plt.text(y_orig[idx, i], y_orig[idx, j], str(idx), color="black", fontsize=8, ha="center", va="center")

            # Draw Pareto front for original data if requested
            if pareto_front_orig:
                sorted_indices_orig = np.argsort(y_orig[pareto_mask_orig, i])
                plt.plot(
                    y_orig[pareto_mask_orig, i][sorted_indices_orig],
                    y_orig[pareto_mask_orig, j][sorted_indices_orig],
                    "k-",
                    alpha=a,
                    label="Original Pareto Front")

        if y_rf is not None:
            # Determine Pareto optimal points for predicted data
            minimize = pareto == "min"
            pareto_mask = is_pareto_efficient(y_rf[:, [i, j]], minimize)

            # Plot all predicted points
            plt.scatter(y_rf[:, i], y_rf[:, j], edgecolor="w", c=y_rf_color, s=s, marker="^", alpha=a, label="Predicted Points")

            # Highlight Pareto points for predicted data
            plt.scatter(y_rf[pareto_mask, i], y_rf[pareto_mask, j], edgecolor="k", c=y_rf_color, s=pareto_size, marker="s", alpha=a, label="Predicted Pareto")

            # Label Pareto points for predicted data if requested
            if pareto_label:
                for idx in np.where(pareto_mask)[0]:
                    plt.text(y_rf[idx, i], y_rf[idx, j], str(idx), color="black", fontsize=8, ha="center", va="center")

            # Draw Pareto front for predicted data if requested
            if pareto_front_y_rf:
                sorted_indices = np.argsort(y_rf[pareto_mask, i])
                plt.plot(
                    y_rf[pareto_mask, i][sorted_indices],
                    y_rf[pareto_mask, j][sorted_indices],
                    linestyle="-",  # Specify the line style
                    color=y_rf_color,  # Use the color specified by y_rf_color
                    alpha=a,
                    label="Predicted Pareto Front",
                )

        # Plot the best point, if provided
        if y_best is not None:
            # Ensure y_best is 2D
            if y_best.ndim == 1:
                y_best = y_best.reshape(1, -1)
            plt.scatter(y_best[:, i], y_best[:, j], edgecolor="k", c=y_best_color, s=s, marker="D", alpha=1, label="Best")

        if target_names is not None:
            plt.xlabel(target_names[i]) 
            plt.ylabel(target_names[j])
        plt.grid()
        plt.title(title)
        plt.legend()
        # plt.show() # Commented out to avoid blocking execution

# Prepare data for plotting
y_best_vals = np.array([[y1_pred, y2_pred, y_mm]])
target_names = ["y1 (min)", "y2 (min)", "y_mm (max)"]
combinations = [(0, 1), (0, 2), (1, 2)]

plot_mo(
    combinations=combinations,
    pareto="min",
    y_rf=callback_values,
    pareto_front_y_rf=True,
    y_best=y_best_vals,
    title="Optimization Trajectory",
    target_names=target_names,
    y_rf_color="blue",
    y_best_color="red"
)
print("Verification script completed successfully.")
