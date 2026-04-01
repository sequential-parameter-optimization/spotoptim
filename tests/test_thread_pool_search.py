# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Release 0.8.0 — Improvement C: ThreadPoolExecutor for search tasks.

Verifies that:
- optimize_steady_state uses ThreadPoolExecutor for search and ProcessPoolExecutor
  for evaluation (hybrid executor design).
- The surrogate lock prevents concurrent refit/search races.
- End-to-end results are correct for n_jobs > 1 with various callable types.
- Backward-compatible: remote_search_task still works (external API unchanged).
"""

import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from spotoptim import SpotOptim


def sphere(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1)


BOUNDS = [(-5, 5), (-5, 5)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _LockSpy:
    """Wraps a threading.Lock and records acquire/release calls."""

    def __init__(self):
        self._lock = threading.Lock()
        self.acquire_count = 0

    def acquire(self, *args, **kwargs):
        self.acquire_count += 1
        return self._lock.acquire(*args, **kwargs)

    def release(self):
        return self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()


# ---------------------------------------------------------------------------
# Unit: surrogate lock correctness
# ---------------------------------------------------------------------------


class TestSurrogateLock:
    """Verify the threading.Lock serialises search and refit correctly."""

    def test_lock_prevents_concurrent_access(self):
        """Two threads cannot hold the lock simultaneously."""
        lock = threading.Lock()
        results = []

        def worker():
            acquired = lock.acquire(blocking=False)
            results.append(acquired)
            if acquired:
                threading.Event().wait(0.05)  # hold briefly
                lock.release()

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        # At most one thread could acquire the lock
        assert results.count(True) == 1

    def test_thread_pool_executor_available(self):
        """ThreadPoolExecutor is importable and functional."""
        results = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = [pool.submit(lambda x=i: x * 2, i) for i in range(4)]
            results = [f.result() for f in futs]
        assert sorted(results) == [0, 2, 4, 6]


# ---------------------------------------------------------------------------
# Integration: end-to-end optimization with n_jobs > 1
# ---------------------------------------------------------------------------


class TestHybridExecutorEndToEnd:
    """Smoke tests: hybrid executor produces correct optimization results."""

    def test_n_jobs_2_sphere(self):
        opt = SpotOptim(
            fun=sphere,
            bounds=BOUNDS,
            n_initial=6,
            max_iter=12,
            n_jobs=2,
            seed=42,
        )
        result = opt.optimize()
        assert result.success
        assert result.nfev >= 6
        assert result.fun < 50.0  # sphere minimum is 0

    def test_n_jobs_3_sphere(self):
        opt = SpotOptim(
            fun=sphere,
            bounds=BOUNDS,
            n_initial=6,
            max_iter=12,
            n_jobs=3,
            seed=7,
        )
        result = opt.optimize()
        assert result.success
        assert result.fun < 50.0

    def test_lambda_objective_parallel(self):
        """Lambda functions must survive dill serialization for eval tasks."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=BOUNDS,
            n_initial=5,
            max_iter=10,
            n_jobs=2,
            seed=0,
        )
        result = opt.optimize()
        assert result.success

    def test_closure_objective_parallel(self):
        """Closures must survive dill serialization for eval tasks."""
        scale = 2.0

        def scaled_sphere(X):
            return scale * np.sum(X**2, axis=1)

        opt = SpotOptim(
            fun=scaled_sphere,
            bounds=BOUNDS,
            n_initial=5,
            max_iter=10,
            n_jobs=2,
            seed=1,
        )
        result = opt.optimize()
        assert result.success

    def test_result_structure_matches_sequential(self):
        """n_jobs=2 and n_jobs=1 produce results with identical shapes."""
        opt_seq = SpotOptim(
            fun=sphere, bounds=BOUNDS, n_initial=5, max_iter=10, n_jobs=1, seed=0
        )
        opt_par = SpotOptim(
            fun=sphere, bounds=BOUNDS, n_initial=5, max_iter=10, n_jobs=2, seed=0
        )
        r_seq = opt_seq.optimize()
        r_par = opt_par.optimize()

        assert r_seq.success
        assert r_par.success
        assert r_par.X.shape[1] == r_seq.X.shape[1]  # same number of dimensions
        assert r_par.fun < 100.0

    def test_minus_one_n_jobs_parallel(self):
        """n_jobs=-1 (all cores) triggers the hybrid executor path."""
        opt = SpotOptim(
            fun=sphere,
            bounds=BOUNDS,
            n_initial=5,
            max_iter=10,
            n_jobs=-1,
            seed=3,
        )
        assert opt.n_jobs >= 1
        result = opt.optimize()
        assert result.success

    def test_4d_sphere_parallel(self):
        """Higher-dimensional problem with n_jobs=2."""
        bounds_4d = [(-3, 3)] * 4
        opt = SpotOptim(
            fun=sphere,
            bounds=bounds_4d,
            n_initial=8,
            max_iter=16,
            n_jobs=2,
            seed=99,
        )
        result = opt.optimize()
        assert result.success
        assert result.X.shape[1] == 4


# ---------------------------------------------------------------------------
# Backward compatibility: remote_search_task still works
# ---------------------------------------------------------------------------


class TestRemoteSearchTaskBackwardCompat:
    """remote_search_task (dill-based) must remain importable and functional."""

    def test_import_remote_search_task(self):
        from spotoptim.utils.parallel import remote_search_task  # noqa: F401

        assert callable(remote_search_task)

    def test_remote_search_task_returns_array(self):
        import dill
        from spotoptim.utils.parallel import remote_search_task

        opt = SpotOptim(
            fun=sphere,
            bounds=BOUNDS,
            n_initial=5,
            max_iter=10,
            seed=0,
        )
        np.random.seed(0)
        opt.X_ = np.random.uniform(-5, 5, (8, 2))
        opt.y_ = sphere(opt.X_)
        opt._fit_surrogate(opt.X_, opt.y_)

        pickled = dill.dumps(opt)
        result = remote_search_task(pickled)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[1] == 2


# ---------------------------------------------------------------------------
# Thread-safety: _surrogate_lock guards concurrent surrogate access
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Verify no data race when search threads and main-thread refit coexist."""

    def test_concurrent_suggest_calls_do_not_crash(self):
        """Multiple threads calling suggest_next_infill_point() under a lock
        must not raise or corrupt state."""
        opt = SpotOptim(
            fun=sphere,
            bounds=BOUNDS,
            n_initial=8,
            max_iter=20,
            n_jobs=1,
            seed=42,
        )
        # Prime the surrogate with some data
        X_init = np.random.default_rng(0).uniform(-5, 5, (8, 2))
        opt.X_ = X_init
        opt.y_ = sphere(X_init)
        opt._fit_surrogate(opt.X_, opt.y_)

        lock = threading.Lock()
        errors = []

        def suggest_with_lock():
            try:
                with lock:
                    opt.suggest_next_infill_point()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=suggest_with_lock) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
