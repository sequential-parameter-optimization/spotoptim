# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Steady-state parallel optimization loop."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Optional, Tuple

import dill
import numpy as np
from scipy.optimize import OptimizeResult

from spotoptim.utils.parallel import (
    is_gil_disabled,
    remote_batch_eval_wrapper,
    remote_eval_wrapper,
)

if TYPE_CHECKING:
    from spotoptim.core.protocol import SpotOptimProtocol


def update_storage_steady(optimizer: SpotOptimProtocol, x, y):
    """Helper to safely append single point (for steady state).

    Args:
        optimizer: SpotOptim instance.
        x (ndarray): New point(s) in original scale.
        y (float or ndarray): Corresponding function value(s).
    """
    x = np.atleast_2d(x)
    if optimizer.X_ is None:
        optimizer.X_ = x
        optimizer.y_ = np.array([y])
    else:
        optimizer.X_ = np.vstack([optimizer.X_, x])
        optimizer.y_ = np.append(optimizer.y_, y)

    # Update best
    if optimizer.best_y_ is None or y < optimizer.best_y_:
        optimizer.best_y_ = y
        optimizer.best_x_ = x.flatten()

    optimizer.min_y = optimizer.best_y_
    optimizer.min_X = optimizer.best_x_


def optimize_steady_state(
    optimizer: SpotOptimProtocol,
    timeout_start: float,
    X0: Optional[np.ndarray],
    y0_known: Optional[float] = None,
    max_iter_override: Optional[int] = None,
) -> Tuple[str, OptimizeResult]:
    """Perform steady-state asynchronous optimization (n_jobs > 1).

    Args:
        optimizer: SpotOptim instance.
        timeout_start (float): Start time for timeout.
        X0 (Optional[np.ndarray]): Initial design points in Natural Space.
        y0_known (Optional[float]): Known best objective value from a previous run.
        max_iter_override (Optional[int]): Override for maximum number of iterations.

    Raises:
        RuntimeError: If all initial design evaluations fail.

    Returns:
        Tuple[str, OptimizeResult]: Tuple containing status and optimization result.
    """
    # Setup similar to _optimize_single_run
    optimizer.set_seed()
    X0 = optimizer.get_initial_design(X0)
    X0 = optimizer.curate_initial_design(X0)

    # Restart injection
    y0_prefilled = np.full(len(X0), np.nan)
    if y0_known is not None and optimizer.x0 is not None:
        dists = np.linalg.norm(X0 - optimizer.x0, axis=1)
        matches = dists < 1e-9
        if np.any(matches):
            if optimizer.verbose:
                print("Skipping re-evaluation of injected best point.")
            y0_prefilled[matches] = y0_known

    effective_max_iter = (
        max_iter_override if max_iter_override is not None else optimizer.max_iter
    )

    from contextlib import ExitStack
    from concurrent.futures import (
        ProcessPoolExecutor,
        ThreadPoolExecutor,
        wait,
        FIRST_COMPLETED,
    )

    _no_gil = is_gil_disabled()
    _surrogate_lock = threading.Lock()

    def _thread_search_task():
        """Search task for ThreadPoolExecutor: direct call, no dill."""
        with _surrogate_lock:
            return optimizer.suggest_next_infill_point()

    def _thread_eval_task_single(x):
        """Thread-based single-point eval for free-threaded Python (no dill)."""
        try:
            x_2d = x.reshape(1, -1)
            y_arr = optimizer.evaluate_function(x_2d)
            return x, y_arr[0]
        except Exception as e:
            return None, e

    def _thread_batch_eval_task(X_batch):
        """Thread-based batch eval for free-threaded Python (no dill)."""
        try:
            y_batch = optimizer.evaluate_function(X_batch)
            return X_batch, y_batch
        except Exception as e:
            return None, e

    with ExitStack() as _stack:
        eval_pool = _stack.enter_context(
            ThreadPoolExecutor(max_workers=optimizer.n_jobs)
            if _no_gil
            else ProcessPoolExecutor(max_workers=optimizer.n_jobs)
        )
        search_pool = _stack.enter_context(
            ThreadPoolExecutor(max_workers=optimizer.n_jobs)
        )
        futures = {}

        # --- Phase 1: Initial Design Evaluation ---
        n_to_submit = 0
        for i, x in enumerate(X0):
            if np.isfinite(y0_prefilled[i]):
                optimizer._update_storage_steady(x, y0_prefilled[i])
                continue
            if _no_gil:
                fut = eval_pool.submit(_thread_eval_task_single, x)
            else:
                _tb_writer_temp = optimizer.tb_writer
                optimizer.tb_writer = None
                try:
                    pickled_args = dill.dumps((optimizer, x))
                finally:
                    optimizer.tb_writer = _tb_writer_temp
                fut = eval_pool.submit(remote_eval_wrapper, pickled_args)
            futures[fut] = "eval"
            n_to_submit += 1

        if optimizer.verbose:
            n_injected = int(np.sum(np.isfinite(y0_prefilled)))
            suffix = (
                f" ({n_injected} injected from restart, skipped re-evaluation)."
                if n_injected
                else "."
            )
            print(
                f"Submitted {n_to_submit} initial points for parallel evaluation{suffix}"
            )

        # Wait for all submitted initial evaluations to complete.
        initial_done_count = 0
        while initial_done_count < n_to_submit:
            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                ftype = futures.pop(fut)
                if ftype != "eval":
                    continue

                try:
                    x_done, y_done = fut.result()
                    if isinstance(y_done, Exception):
                        if optimizer.verbose:
                            print(f"Eval failed: {y_done}")
                    else:
                        optimizer._update_storage_steady(x_done, y_done)
                except Exception as e:
                    if optimizer.verbose:
                        print(f"Task failed: {e}")

                initial_done_count += 1

        # Init tensorboard and stats
        optimizer._init_tensorboard()

        if optimizer.y_ is None or len(optimizer.y_) == 0:
            raise RuntimeError(
                "All initial design evaluations failed. "
                "Check your objective function for pickling issues or missing imports "
                "(e.g. numpy) in the worker process. "
                "If defining functions in a notebook/script, ensure imports are inside "
                "the function."
            )

        optimizer.update_stats()
        optimizer.get_best_xy_initial_design()

        # Fit first surrogate (no lock needed — no search threads active yet)
        if optimizer.verbose:
            print(
                f"Initial design evaluated. Fitting surrogate... "
                f"(Data size: {len(optimizer.y_)})"
            )
        optimizer.fit_scheduler()

        # --- Phase 2: Steady State Loop ---
        if optimizer.verbose:
            print("Starting steady-state optimization loop...")

        pending_cands: list = []
        _future_n_pts: dict = {}

        def _flush_batch() -> None:
            """Dispatch all pending_cands as a single batch eval task."""
            nonlocal pending_cands
            X_batch = np.vstack(pending_cands)
            n_in_batch = len(pending_cands)
            pending_cands = []
            if _no_gil:
                fut_eval = eval_pool.submit(_thread_batch_eval_task, X_batch)
            else:
                _tb_writer_temp = optimizer.tb_writer
                optimizer.tb_writer = None
                try:
                    pickled_args = dill.dumps((optimizer, X_batch))
                finally:
                    optimizer.tb_writer = _tb_writer_temp
                fut_eval = eval_pool.submit(remote_batch_eval_wrapper, pickled_args)
            futures[fut_eval] = "batch_eval"
            _future_n_pts[fut_eval] = n_in_batch

        def _batch_ready() -> bool:
            """True when pending_cands should be flushed to eval_pool."""
            if not pending_cands:
                return False
            if len(pending_cands) >= optimizer.eval_batch_size:
                return True
            return not any(t == "search" for t in futures.values())

        while (len(optimizer.y_) < effective_max_iter) and (
            time.time() < timeout_start + optimizer.max_time * 60
        ):
            if _batch_ready():
                _flush_batch()

            n_active = len(futures)
            n_slots = optimizer.n_jobs - n_active

            if n_slots > 0:
                for _ in range(n_slots):
                    n_in_flight = sum(_future_n_pts.values())
                    n_searches = sum(1 for t in futures.values() if t == "search")
                    reserved = (
                        len(optimizer.y_)
                        + n_in_flight
                        + n_searches
                        + len(pending_cands)
                    )
                    if reserved < effective_max_iter:
                        fut = search_pool.submit(_thread_search_task)
                        futures[fut] = "search"
                    else:
                        break

            if _batch_ready():
                _flush_batch()

            if not futures and not pending_cands:
                break

            if not futures:
                _flush_batch()
                continue

            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                ftype = futures.pop(fut)
                try:
                    res = fut.result()
                    if isinstance(res, Exception):
                        if optimizer.verbose:
                            print(f"Remote {ftype} failed: {res}")
                        _future_n_pts.pop(fut, None)
                        continue

                    if ftype == "search":
                        x_cand = res
                        pending_cands.append(x_cand)
                        if _batch_ready():
                            _flush_batch()

                    elif ftype == "batch_eval":
                        _future_n_pts.pop(fut, None)
                        X_done, y_done = res
                        if isinstance(y_done, Exception):
                            if optimizer.verbose:
                                print(f"Batch eval failed: {y_done}")
                        else:
                            for xi, yi in zip(X_done, y_done):
                                optimizer.update_success_rate(np.array([yi]))
                                optimizer._update_storage_steady(xi, yi)
                                optimizer.n_iter_ += 1

                            if optimizer.verbose:
                                if optimizer.max_time != np.inf:
                                    prog_val = (
                                        (time.time() - timeout_start)
                                        / (optimizer.max_time * 60)
                                        * 100
                                    )
                                    progress_str = f"Time: {prog_val:.1f}%"
                                else:
                                    prog_val = (
                                        len(optimizer.y_) / effective_max_iter * 100
                                    )
                                    progress_str = f"Evals: {prog_val:.1f}%"

                                print(
                                    f"Iter {len(optimizer.y_)}/{effective_max_iter}"
                                    f" | Best: {optimizer.best_y_:.6f}"
                                    f" | Rate: {optimizer.success_rate:.2f}"
                                    f" | {progress_str}"
                                )

                            with _surrogate_lock:
                                optimizer.fit_scheduler()

                except Exception as e:
                    _future_n_pts.pop(fut, None)
                    if optimizer.verbose:
                        print(f"Error processing future: {e}")

    return "FINISHED", OptimizeResult(
        x=optimizer.best_x_,
        fun=optimizer.best_y_,
        nfev=len(optimizer.y_),
        nit=optimizer.n_iter_,
        success=True,
        message="Optimization finished (Steady State)",
        X=optimizer.X_,
        y=optimizer.y_,
    )
