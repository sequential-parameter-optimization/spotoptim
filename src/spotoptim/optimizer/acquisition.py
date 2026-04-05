# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Acquisition function optimization and infill point selection."""

from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import OptimizeResult, differential_evolution, minimize
from scipy.spatial.distance import cdist

from spotoptim.sampling.design import generate_uniform_design
from spotoptim.tricands import tricands


def optimize_acquisition_tricands(optimizer) -> np.ndarray:
    """Optimize using geometric infill strategy via triangulation candidates.

    Args:
        optimizer: SpotOptim instance.

    Returns:
        ndarray: The optimized point(s).
    """
    if (
        not hasattr(optimizer, "X_")
        or optimizer.X_ is None
        or len(optimizer.X_) < optimizer.n_dim + 1
    ):
        n_design = max(1, optimizer.acquisition_fun_return_size)
        x0 = generate_uniform_design(optimizer.bounds, n_design, seed=optimizer.rng)

        if optimizer.acquisition_fun_return_size <= 1:
            return x0.flatten()
        return x0

    # Generate candidates
    nmax = max(100 * optimizer.n_dim, optimizer.acquisition_fun_return_size * 50)

    # Normalize X_ to [0, 1] relative to bounds
    X_norm = (optimizer.X_ - optimizer.lower) / (optimizer.upper - optimizer.lower)

    # Generate candidates in [0, 1] space
    X_cands_norm = tricands(
        X_norm,
        nmax=nmax,
        lower=0.0,
        upper=1.0,
        fringe=optimizer.tricands_fringe,
    )

    # Denormalize candidates back to original space
    X_cands = X_cands_norm * (optimizer.upper - optimizer.lower) + optimizer.lower

    # Evaluate acquisition function on all candidates
    acq_values = optimizer._acquisition_function(X_cands.T)

    # Sort indices (smallest is best because of negation)
    sorted_indices = np.argsort(acq_values)

    # Select top n
    top_n = min(optimizer.acquisition_fun_return_size, len(sorted_indices))
    best_indices = sorted_indices[:top_n]
    return X_cands[best_indices]


def prepare_de_kwargs(optimizer, x0=None):
    """Prepare kwargs for differential_evolution, extracting options if necessary.

    Args:
        optimizer: SpotOptim instance.
        x0: Initial point (unused, kept for API compatibility).

    Returns:
        dict: Filtered kwargs for differential_evolution.
    """
    kwargs = (optimizer.config.acquisition_optimizer_kwargs or {}).copy()

    # Extract 'options' if present (compatibility with minimize structure)
    if "options" in kwargs and isinstance(kwargs["options"], dict):
        options = kwargs.pop("options")
        kwargs.update(options)

    # Define valid arguments for differential_evolution
    valid_de_args = {
        "strategy",
        "maxiter",
        "popsize",
        "tol",
        "mutation",
        "recombination",
        "seed",
        "callback",
        "disp",
        "polish",
        "init",
        "atol",
        "updating",
        "workers",
        "constraints",
        "x0",
    }

    # Filter kwargs to only include valid DE arguments
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_de_args}

    # Set defaults if not provided
    if "maxiter" not in filtered_kwargs:
        filtered_kwargs["maxiter"] = 1000

    return filtered_kwargs


def optimize_acquisition_de(optimizer) -> np.ndarray:
    """Optimize using differential evolution.

    Args:
        optimizer: SpotOptim instance.

    Returns:
        ndarray: The optimized point(s).
    """
    # Variables to capture population from callback
    population = None
    population_energies = None

    # Determine which "best" to use
    if (optimizer.repeats_initial > 1 or optimizer.repeats_surrogate > 1) and hasattr(
        optimizer, "min_mean_X"
    ):
        best_x = optimizer.min_mean_X
    else:
        best_x = optimizer.best_x_

    if best_x is not None:
        best_x = optimizer.transform_X(best_x)
        # Nudge x0 strictly inside DE bounds using nextafter
        _lb = np.array([b[0] for b in optimizer.bounds])
        _ub = np.array([b[1] for b in optimizer.bounds])
        best_x = np.clip(best_x, np.nextafter(_lb, _ub), np.nextafter(_ub, _lb))
        best_X = best_x if optimizer.rng.rand() < optimizer.de_x0_prob else None
    else:
        best_X = None

    def callback(intermediate_result: OptimizeResult):
        nonlocal population, population_energies
        if hasattr(intermediate_result, "population"):
            population = intermediate_result.population
            population_energies = intermediate_result.population_energies

    result = differential_evolution(
        func=optimizer._acquisition_function,
        bounds=optimizer.bounds,
        seed=optimizer.rng,
        callback=callback,
        x0=best_X,
        vectorized=True,
        **prepare_de_kwargs(optimizer, best_X),
    )

    if optimizer.acquisition_fun_return_size > 1:
        if population is not None and population_energies is not None:
            sorted_indices = np.argsort(population_energies)
            top_n = min(optimizer.acquisition_fun_return_size, len(sorted_indices))

            candidates = [result.x]

            if top_n > 1:
                next_indices = sorted_indices[1:top_n]
                candidates.extend(population[next_indices])

            return np.array(candidates)
        else:
            return result.x.reshape(1, -1)

    return result.x


def optimize_acquisition_scipy(optimizer) -> np.ndarray:
    """Optimize using scipy.optimize.minimize interface (default).

    Args:
        optimizer: SpotOptim instance.

    Returns:
        ndarray: The optimized acquisition function values.

    Raises:
        ValueError: If acquisition optimizer is not a string or callable.
    """
    # Generate random x0 within bounds
    low = np.array([b[0] for b in optimizer.bounds])
    high = np.array([b[1] for b in optimizer.bounds])
    x0 = optimizer.rng.uniform(low, high)

    if isinstance(optimizer.acquisition_optimizer, str):
        kwargs = optimizer.config.acquisition_optimizer_kwargs or {}

        run_kwargs = kwargs.copy()
        if "method" not in run_kwargs:
            run_kwargs["method"] = optimizer.acquisition_optimizer

        # Define valid arguments for minimize()
        valid_minimize_args = {
            "args",
            "method",
            "jac",
            "hess",
            "hessp",
            "constraints",
            "tol",
            "callback",
            "options",
        }

        # Move any argument that is NOT a valid minimize() argument into 'options'
        if "options" not in run_kwargs:
            run_kwargs["options"] = {}

        keys_to_move = [k for k in run_kwargs if k not in valid_minimize_args]
        for k in keys_to_move:
            run_kwargs["options"][k] = run_kwargs.pop(k)

        result = minimize(
            fun=optimizer._acquisition_function,
            x0=x0,
            bounds=optimizer.bounds,
            **run_kwargs,
        )
    elif callable(optimizer.acquisition_optimizer):
        result = optimizer.acquisition_optimizer(
            fun=optimizer._acquisition_function, x0=x0, bounds=optimizer.bounds
        )
    else:
        raise ValueError(
            f"Unknown acquisition optimizer type: {type(optimizer.acquisition_optimizer)}"
        )

    if optimizer.acquisition_fun_return_size > 1:
        return result.x.reshape(1, -1)

    return result.x


def try_optimizer_candidates(
    optimizer, n_needed: int = 1, current_batch: Optional[List[np.ndarray]] = None
) -> List[np.ndarray]:
    """Try candidates proposed by the acquisition result optimizer.

    Args:
        optimizer: SpotOptim instance.
        n_needed (int): Number of candidates needed.
        current_batch (list): Points already selected in current iteration.

    Returns:
        List[ndarray]: List of unique valid candidate points found.
    """
    valid_candidates = []
    if current_batch is None:
        current_batch = []

    x_next_candidates = optimizer.optimize_acquisition_func()

    # Ensure iterable of 1D arrays
    if x_next_candidates.ndim == 1:
        obs_candidates = [x_next_candidates]
    else:
        obs_candidates = [
            x_next_candidates[i] for i in range(x_next_candidates.shape[0])
        ]

    X_transformed = optimizer.transform_X(optimizer.X_)

    # Helper to check if a point is valid
    def is_valid(p, reference_set):
        p_rounded = optimizer.repair_non_numeric(p.reshape(1, -1), optimizer.var_type)[
            0
        ]
        p_2d = p_rounded.reshape(1, -1)
        x_new, _ = optimizer.select_new(
            A=p_2d, X=reference_set, tolerance=optimizer.tolerance_x
        )
        return p_rounded if x_new.shape[0] > 0 else None

    for i, x_next in enumerate(obs_candidates):
        if len(valid_candidates) >= n_needed:
            break

        ref_list = [X_transformed]
        if current_batch:
            ref_list.append(np.array(current_batch))
        if valid_candidates:
            ref_list.append(np.array(valid_candidates))

        if len(ref_list) > 1:
            reference_set = np.vstack(ref_list)
        else:
            reference_set = ref_list[0]

        candidate = is_valid(x_next, reference_set)

        if candidate is not None:
            valid_candidates.append(candidate)
        elif optimizer.verbose:
            print(
                f"Optimizer candidate {i + 1}/{len(obs_candidates)} was duplicate/invalid."
            )

    return valid_candidates


def remove_nan(
    optimizer, X: np.ndarray, y: np.ndarray, stop_on_zero_return: bool = True
) -> tuple:
    """Remove rows where y contains NaN or inf values.

    Args:
        optimizer: SpotOptim instance.
        X (ndarray): Design matrix, shape (n_samples, n_features).
        y (ndarray): Objective values, shape (n_samples,).
        stop_on_zero_return (bool): If True, raise error when all values are removed.

    Returns:
        tuple: (X_clean, y_clean) with NaN/inf rows removed.

    Raises:
        ValueError: If all values are NaN/inf and stop_on_zero_return is True.
    """
    finite_mask = np.isfinite(y)

    if not np.any(finite_mask):
        msg = "All objective function values are NaN or inf."
        if stop_on_zero_return:
            raise ValueError(msg)
        else:
            if optimizer.verbose:
                print(f"Warning: {msg} Returning empty arrays.")
            return np.array([]).reshape(0, X.shape[1]), np.array([])

    n_removed = np.sum(~finite_mask)
    if n_removed > 0 and optimizer.verbose:
        print(f"Warning: Removed {n_removed} sample(s) with NaN/inf values")

    return X[finite_mask], y[finite_mask]


def handle_acquisition_failure(optimizer) -> np.ndarray:
    """Handle acquisition failure by proposing new design points.

    Args:
        optimizer: SpotOptim instance.

    Returns:
        ndarray: New design point as a fallback, shape (n_features,).
    """
    if optimizer.acquisition_failure_strategy == "random":
        if optimizer.verbose:
            print("Acquisition failure: Using random space-filling design as fallback.")
        x_new_unit = optimizer.lhs_sampler.random(n=1)[0]
        x_new = optimizer.lower + x_new_unit * (optimizer.upper - optimizer.lower)

    return optimizer.repair_non_numeric(x_new.reshape(1, -1), optimizer.var_type)[0]


def try_fallback_strategy(
    optimizer,
    max_attempts: int = 10,
    current_batch: Optional[List[np.ndarray]] = None,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Try fallback strategy (e.g. random search) to find a unique point.

    Args:
        optimizer: SpotOptim instance.
        max_attempts (int): Maximum number of fallback attempts.
        current_batch (list): Points already selected in current iteration.

    Returns:
        Tuple[Optional[ndarray], ndarray]:
            - The first element is the unique valid candidate point if found, else None.
            - The second element is the last attempted point.
    """
    x_last = None
    if current_batch is None:
        current_batch = []

    X_transformed = optimizer.transform_X(optimizer.X_)

    for attempt in range(max_attempts):
        if optimizer.verbose:
            print(
                f"Fallback attempt {attempt + 1}/{max_attempts}: Using fallback strategy"
            )
        x_next = optimizer._handle_acquisition_failure()

        x_next_rounded = optimizer.repair_non_numeric(
            x_next.reshape(1, -1), optimizer.var_type
        )[0]
        x_last = x_next_rounded

        x_next_2d = x_next_rounded.reshape(1, -1)

        ref_list = [X_transformed]
        if len(current_batch) > 0:
            ref_list.append(np.array(current_batch))

        if len(ref_list) > 1:
            reference_set = np.vstack(ref_list)
        else:
            reference_set = ref_list[0]

        x_new, _ = optimizer.select_new(
            A=x_next_2d, X=reference_set, tolerance=optimizer.tolerance_x
        )

        if x_new.shape[0] > 0:
            return x_next_rounded, x_last

    return None, x_last


def optimize_acquisition_func(optimizer) -> np.ndarray:
    """Optimize the acquisition function to find the next point to evaluate.

    Args:
        optimizer: SpotOptim instance.

    Returns:
        ndarray: The optimized point(s).
    """
    if optimizer.acquisition_optimizer == "tricands":
        return optimizer._optimize_acquisition_tricands()
    elif optimizer.acquisition_optimizer == "differential_evolution":
        return optimizer._optimize_acquisition_de()
    elif optimizer.acquisition_optimizer == "de_tricands":
        val = optimizer.rng.rand()
        if val < optimizer.prob_de_tricands:
            return optimizer._optimize_acquisition_de()
        else:
            return optimizer._optimize_acquisition_tricands()
    else:
        return optimizer._optimize_acquisition_scipy()


def select_new(
    optimizer, A: np.ndarray, X: np.ndarray, tolerance: float = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Select rows from A that are not in X.

    Args:
        optimizer: SpotOptim instance.
        A (ndarray): Array with new values.
        X (ndarray): Array with known values.
        tolerance (float): Tolerance value for comparison. Defaults to 0.

    Returns:
        tuple: (new_values, is_new_mask).
    """
    if len(X) == 0:
        return A, np.ones(len(A), dtype=bool)

    dists = cdist(A, X, metric=optimizer.min_tol_metric)
    is_duplicate = np.any(dists <= tolerance, axis=1)
    ind = is_duplicate
    return A[~ind], ~ind


def suggest_next_infill_point(optimizer) -> np.ndarray:
    """Suggest next point to evaluate (dispatcher).

    Args:
        optimizer: SpotOptim instance.

    Returns:
        ndarray: Next point(s) to evaluate in Transformed and Mapped Space.
            Shape is (n_infill_points, n_features).
    """
    # 1. Optimizer candidates
    candidates = []
    opt_candidates = optimizer._try_optimizer_candidates(
        n_needed=optimizer.n_infill_points, current_batch=candidates
    )
    candidates.extend(opt_candidates)

    if len(candidates) >= optimizer.n_infill_points:
        return np.vstack(candidates)

    # 2. Try fallback strategy to fill remaining slots
    while len(candidates) < optimizer.n_infill_points:
        cand, x_last = optimizer._try_fallback_strategy(
            max_attempts=10, current_batch=candidates
        )
        if cand is not None:
            candidates.append(cand)
        else:
            if optimizer.verbose:
                print(
                    "Warning: Could not fill all infill points with unique candidates."
                )
            break

    if len(candidates) > 0:
        return np.vstack(candidates)

    # 3. Return last attempt (duplicate) if absolutely nothing found
    if optimizer.verbose:
        print(
            "Warning: Could not find unique point after optimization candidates "
            "and fallback attempts. Returning last candidate (duplicate)."
        )

    if x_last is None:
        x_next = optimizer._handle_acquisition_failure()
        return x_next.reshape(1, -1)

    return x_last.reshape(1, -1)
