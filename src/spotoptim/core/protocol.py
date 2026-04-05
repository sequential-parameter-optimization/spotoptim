# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Structural typing protocol for the SpotOptim optimizer.

Extracted module functions accept ``optimizer: SpotOptimProtocol`` instead of
``optimizer: object`` so that *ty*, mypy, and other type checkers can verify
attribute access without importing the concrete ``SpotOptim`` class (which
would create circular imports).

The protocol deliberately uses *runtime_checkable=False* (the default) —
it is a static-analysis-only contract.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.random import RandomState
from typing_extensions import Protocol


class SpotOptimProtocol(Protocol):
    """Structural type describing the optimizer interface used by extracted modules.

    Every attribute and method listed here is accessed by at least one module-level
    function in ``spotoptim.utils``, ``spotoptim.reporting``, ``spotoptim.optimizer``,
    ``spotoptim.core.storage``, or ``spotoptim.plot``.
    """

    # ------------------------------------------------------------------
    # Configuration (from SpotOptimConfig / __init__)
    # ------------------------------------------------------------------
    bounds: Optional[list]
    max_iter: int
    n_initial: int
    surrogate: Optional[object]
    acquisition: str
    var_type: Optional[list]
    var_name: Optional[list]
    var_trans: Optional[list]
    tolerance_x: Optional[float]
    max_time: float
    repeats_initial: int
    repeats_surrogate: int
    ocba_delta: int
    tensorboard_log: bool
    tensorboard_path: Optional[str]
    tensorboard_clean: bool
    fun_mo2so: Optional[Callable]
    seed: Optional[int]
    verbose: bool
    n_infill_points: int
    max_surrogate_points: Optional[Union[int, List[int]]]
    selection_method: str
    acquisition_failure_strategy: str
    penalty: bool
    penalty_val: Optional[float]
    acquisition_fun_return_size: int
    acquisition_optimizer: Union[str, Callable]
    restart_after_n: int
    restart_inject_best: bool
    x0: Optional[np.ndarray]
    de_x0_prob: float
    tricands_fringe: bool
    prob_de_tricands: float
    window_size: Optional[int]
    min_tol_metric: str
    prob_surrogate: Optional[List[float]]
    n_jobs: int
    eval_batch_size: int
    acquisition_optimizer_kwargs: Optional[Dict[str, Any]]
    args: Tuple
    kwargs: Optional[Dict[str, Any]]
    config: Any  # SpotOptimConfig dataclass instance

    # ------------------------------------------------------------------
    # Instance attributes set in __init__
    # ------------------------------------------------------------------
    fun: Callable
    eps: float
    n_dim: int
    lower: np.ndarray
    upper: np.ndarray
    _original_lower: np.ndarray
    _original_upper: np.ndarray
    _factor_maps: Dict[int, Dict[int, str]]
    rng: RandomState
    lhs_sampler: Any  # scipy.stats.qmc.LatinHypercube
    objective_names: Optional[List[str]]
    model: Optional[object]  # fitted surrogate reference
    tb_writer: Any  # Optional[SummaryWriter]

    # Dimension reduction
    red_dim: bool
    ident: Optional[List[bool]]
    all_lower: np.ndarray
    all_upper: np.ndarray
    all_var_name: Optional[list]
    all_var_type: Optional[list]
    all_var_trans: Optional[list]

    # ------------------------------------------------------------------
    # Mutable state (from SpotOptimState)
    # ------------------------------------------------------------------
    X_: Optional[np.ndarray]
    y_: Optional[np.ndarray]
    y_mo: Optional[np.ndarray]
    best_x_: Optional[np.ndarray]
    best_y_: Optional[float]
    n_iter_: int
    counter: int
    success_rate: float
    _success_history: List
    mean_X: Optional[np.ndarray]
    mean_y: Optional[np.ndarray]
    var_y: Optional[np.ndarray]
    min_mean_X: Optional[np.ndarray]
    min_mean_y: Optional[float]
    min_var_y: Optional[float]
    min_X: Optional[np.ndarray]
    min_y: Optional[float]

    # ------------------------------------------------------------------
    # Internal surrogates list (multi-surrogate scheduling)
    # ------------------------------------------------------------------
    _surrogates_list: Optional[list]
    _prob_surrogate: Optional[List[float]]

    # ------------------------------------------------------------------
    # Methods called by extracted modules
    # ------------------------------------------------------------------
    def set_seed(self) -> None: ...
    def get_initial_design(self, X0: Optional[np.ndarray]) -> np.ndarray: ...
    def curate_initial_design(self, X0: np.ndarray) -> np.ndarray: ...
    def init_surrogate(self) -> None: ...
    def fit_scheduler(self) -> None: ...
    def evaluate_function(self, X: np.ndarray) -> np.ndarray: ...
    def get_best_xy_initial_design(self) -> None: ...
    def suggest_next_infill_point(self) -> np.ndarray: ...

    def transform_X(self, X: np.ndarray) -> np.ndarray: ...
    def inverse_transform_X(self, X: np.ndarray) -> np.ndarray: ...
    def repair_non_numeric(self, X: np.ndarray) -> np.ndarray: ...
    def map_to_factor_values(self, X: np.ndarray) -> np.ndarray: ...
    def to_all_dim(self, X: np.ndarray) -> np.ndarray: ...

    def select_new(
        self, X_cand: np.ndarray, tolerance_x: Optional[float] = ...
    ) -> np.ndarray: ...
    def optimize_acquisition_func(self) -> np.ndarray: ...
    def _optimize_acquisition_tricands(self) -> np.ndarray: ...
    def _optimize_acquisition_de(self) -> np.ndarray: ...
    def _optimize_acquisition_scipy(self) -> np.ndarray: ...

    def _try_optimizer_candidates(
        self, candidates: np.ndarray
    ) -> Optional[np.ndarray]: ...

    def _try_fallback_strategy(self) -> Optional[np.ndarray]: ...
    def _handle_acquisition_failure(self) -> np.ndarray: ...

    def _acquisition_function(self, X: np.ndarray) -> np.ndarray: ...

    def _predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]: ...

    def update_stats(self) -> None: ...
    def update_success_rate(self, y_new: np.ndarray) -> None: ...
    def _update_storage_steady(self, x: np.ndarray, y: float) -> None: ...

    def _init_tensorboard(self) -> None: ...
    def _close_and_del_tensorboard_writer(self) -> None: ...

    def get_importance(self) -> Any: ...
    def get_stars(self, importance: float) -> str: ...
