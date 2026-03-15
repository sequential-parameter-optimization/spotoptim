# Parallelization Strategy for spotoptim

**Document version:** 2026-03-15
**Applies to:** spotoptim ≥ 0.6.0
**Implementation status:** Improvements E (0.7.0) and C (0.8.0) are complete.

---

## 1. Current Implementation (0.6.0)

### 1.1 Overview

spotoptim implements **steady-state asynchronous Bayesian Optimization (BO)**
using `concurrent.futures.ProcessPoolExecutor`.
The entry point is the `n_jobs` constructor parameter (default `1`).

```
n_jobs == 1  →  _optimize_sequential_run()   (single process, synchronous)
n_jobs  > 1  →  _optimize_steady_state()     (multi-process, asynchronous)
```

### 1.2 Steady-State Architecture

When `n_jobs > 1` the optimizer maintains a live pool of at most `n_jobs`
concurrent futures, split into two task types:

| Task type | What it does | Submitted by |
|-----------|-------------|--------------|
| `search`  | Serializes the entire SpotOptim instance with `dill`, deserializes it in a worker process, calls `suggest_next_infill_point()` | Main process after every surrogate refit |
| `eval`    | Serializes `(optimizer, x)` with `dill`, evaluates the objective function `fun(x)` in a worker process | Main process immediately after a `search` result arrives |

The main loop uses `concurrent.futures.wait(..., return_when=FIRST_COMPLETED)`,
so new tasks are dispatched as soon as any slot frees — there is no synchronous
barrier between generations.

```
Main process
│
├─ ProcessPoolExecutor (n_jobs workers)
│   ├─ Worker 0: search task  → returns x_cand_0
│   ├─ Worker 1: eval task    → returns (x, y)
│   └─ Worker 2: search task  → returns x_cand_2
│
│  On FIRST_COMPLETED:
│  ├─ If eval done  → update X_, y_, refit surrogate, dispatch new search
│  └─ If search done → dispatch eval(x_cand)
```

### 1.3 Why `dill` Instead of `pickle`

The standard `multiprocessing` and `concurrent.futures` serialization uses
`pickle`, which cannot handle lambda functions, local functions, or closures.
spotoptim accepts any callable `fun`, so `dill` is used to serialize the
full optimizer object (including the user-supplied `fun`) to each worker.

### 1.4 Where `differential_evolution` Parallelism Is **Not** Used

SciPy's `differential_evolution` (the default acquisition optimizer) supports a
`workers=` parameter that distributes population evaluation across
`multiprocessing.Pool` workers.  spotoptim does **not** use this parameter:

```python
result = differential_evolution(
    func=self._acquisition_function,
    bounds=self.bounds,
    vectorized=True,   # ← 18× batch speedup via NumPy; used instead of workers=
    ...
)
```

`workers > 1` in SciPy overrides `vectorized=True`, discarding the batch
speedup that is already in place.  Because the acquisition function is cheap
surrogate prediction (NumPy/BLAS), process-spawn overhead would outweigh any
parallelism benefit.

---

## 2. Comparison with SciPy and Python `multiprocessing`

### 2.1 Level of Parallelism

The two systems parallelize at **different levels**:

```
BO outer loop  [spotoptim n_jobs= parallelizes here]
│
└─ BO iteration
   ├─ Fit surrogate
   └─ Optimize acquisition function
      └─ DE inner loop  [SciPy workers= would parallelize here]
         └─ Evaluate pop member (surrogate.predict)
```

spotoptim's `n_jobs` targets the **expensive** part (objective function
evaluations) across BO iterations; SciPy's `workers` targets **cheap**
surrogate predictions within a single DE run.

### 2.2 Mechanism Comparison

| Dimension | SciPy DE `workers=` | spotoptim `n_jobs=` |
|-----------|--------------------|--------------------|
| Executor | `multiprocessing.Pool.map()` | `concurrent.futures.ProcessPoolExecutor` |
| Serialization | `pickle` | `dill` |
| Blocking model | Synchronous (`map` waits for all) | Asynchronous (`FIRST_COMPLETED`) |
| Parallelizes | Population evaluation (inner) | Objective evaluation (outer) |
| Surrogate updates | After each DE generation | After each objective evaluation |
| Callable constraint | Must be `pickle`-able | Any callable (lambdas, closures) |

### 2.3 Strengths of the Current spotoptim Approach

1. **Right level for expensive objectives.** BO is used when `fun` is slow
   (seconds to hours per call). Parallelizing evaluations is where the
   wall-clock gain is.
2. **Asynchronous steady-state.** No worker idles waiting for the slowest
   peer in a synchronous batch.
3. **Continuous surrogate updates.** Every completed evaluation immediately
   refits the model, so subsequent searches use fresher information than a
   fully-synchronous scheme would provide.
4. **`dill` serialization.** Users can pass lambdas and local functions as
   `fun` without restrictions.

---

## 3. Identified Weaknesses and the Improvement Roadmap

The following weaknesses were identified.  Two proposed improvements
(shared memory and SciPy `workers=`) were **discarded** because they conflict
with existing features or are superseded by better alternatives — see
[Section 4](#4-discarded-improvements).

Four compatible improvements are retained and ordered by
dependency and risk:

| # | Improvement | Release | Depends on | Status |
|---|-------------|---------|------------|--------|
| E | `n_jobs=-1` convention | 0.7.0 | — | ✅ Done |
| C | ThreadPoolExecutor for `search` tasks | 0.8.0 | E | ✅ Done |
| F | Batch evaluation API | 0.9.0 | C | Planned |
| D | Free-threaded (no-GIL) awareness | 0.10.0 | C | Planned |

---

## 4. Discarded Improvements

### 4.1 A — Shared Memory for `X_`, `y_` (Superseded by C)

**Idea:** Use `multiprocessing.shared_memory.SharedMemory` to expose the design
matrix `X_` and target vector `y_` without copying them to each worker process.

**Why discarded:** Improvement C (ThreadPoolExecutor for search tasks) makes
this unnecessary.  Threads share the process heap automatically; there is no
copy of `X_` or `y_` across process boundaries for search tasks.  Eval tasks
receive only a single point `x`, never the full dataset.  Implementing
shared memory alongside threads would add complexity with no remaining benefit.

### 4.2 B — `workers=` for `differential_evolution` (Conflicts with `vectorized=True`)

**Idea:** Pass `workers=n_jobs` to `scipy.optimize.differential_evolution`
to parallelize population evaluation within the acquisition optimizer.

**Why discarded:** SciPy explicitly overrides `vectorized=True` when
`workers != 1` (see SciPy docs).  The current 18× batch speedup from
`vectorized=True` would be lost.  Because the acquisition function is cheap
surrogate prediction (NumPy/BLAS operations), process-spawn and IPC overhead
would dominate any parallelism benefit.  The correct place to add parallelism
is the outer objective evaluation, not the inner surrogate query.

---

## 5. Release Roadmap

### Release 0.7.0 — Improvement E: `n_jobs=-1` Convention ✅

**Current problem:** spotoptim does not follow the scikit-learn / SciPy
convention that `n_jobs=-1` means "use all available CPU cores".  Users who
set `n_jobs=-1` currently get `ProcessPoolExecutor(max_workers=-1)`, which
raises a `ValueError`.

**Change:** Resolve `n_jobs` to `os.cpu_count()` (or a safe fallback of `1`)
when `-1` is passed, before the executor is constructed:

```python
import os

def _resolve_n_jobs(n_jobs: int) -> int:
    if n_jobs == -1:
        return os.cpu_count() or 1
    if n_jobs < -1 or n_jobs == 0:
        raise ValueError(f"n_jobs must be a positive integer or -1, got {n_jobs}.")
    return n_jobs
```

**Compatibility:** Purely additive.  No existing behaviour changes when
`n_jobs >= 1`.

**Risk:** Minimal.

---

### Release 0.8.0 — Improvement C: ThreadPoolExecutor for `search` Tasks ✅

**Current problem:** Every `search` task (optimizing the acquisition function)
requires:

1. `dill.dumps(self)` — serializing the entire SpotOptim instance (surrogate
   model, `X_`, `y_`, config, …) into a byte string
2. IPC to a worker process
3. `dill.loads(pickled_optimizer)` — full deserialization in the worker
4. `suggest_next_infill_point()` — the actual work
5. Serialization of the result back to the main process

Steps 1–3 and 5 are pure overhead.  For complex surrogates (e.g. MLPSurrogate
with PyTorch weights) the serialized object can be several megabytes, adding
tens to hundreds of milliseconds **per search task**.

**Key insight:** The search task is **not** CPU-bound in the way that justifies
process isolation.  It calls surrogate model prediction (NumPy matrix
operations, PyTorch inference), which releases the GIL and benefits from
shared L2/L3 cache.  Threads are the correct primitive here.

**Change:** Use a *hybrid* executor design:

```python
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED

with ProcessPoolExecutor(max_workers=n_jobs) as eval_pool, \
     ThreadPoolExecutor(max_workers=n_jobs) as search_pool:

    # eval tasks  → eval_pool   (process isolation; objective fun may be arbitrary)
    # search tasks → search_pool (thread; surrogate shared from main process heap)
    ...
```

The `search` task no longer needs `dill` serialization at all — the worker
thread sees the same Python objects as the main thread:

```python
# Before (process-based, requires full serialization)
pickled_opt = dill.dumps(self)
fut = eval_pool.submit(remote_search_task, pickled_opt)

# After (thread-based, zero copy)
fut = search_pool.submit(self.suggest_next_infill_point)
```

**`eval` tasks** remain in the `ProcessPoolExecutor` because:

- The user's objective `fun` may be arbitrary (could hold the GIL, call
  subprocesses, or have side-effects that require isolation)
- Process isolation prevents a crashing eval from killing the main process

**Expected benefit:** Eliminates the dominant serialization bottleneck for
search tasks.  The surrogate model and all data are shared by reference.
Memory footprint per active search task drops from O(surrogate size) to O(1).

**Compatibility:** The public API (`n_jobs`, `optimize()`) does not change.
The internal `_optimize_steady_state` method is refactored.

**Risk:** Low.  Thread safety must be verified for surrogate `predict()` calls
made concurrently.  sklearn's `GaussianProcessRegressor.predict` is
thread-safe for read-only inference.  PyTorch `model.forward()` with
`torch.no_grad()` is thread-safe.  The main-thread surrogate `fit()` must not
run concurrently with a `search` thread's `predict()`; this is already
guaranteed by the steady-state design (refit happens only after an `eval`
completes, at which point no search task is in flight for the stale model).

---

### Release 0.9.0 — Improvement F: Batch Evaluation API

**Current problem:** Each `eval` task evaluates a single point `x` in its own
worker process.  For every evaluation:

- One process slot is occupied by one objective call
- Process-spawn + IPC overhead is paid per point
- If `fun` supports batch input (as it does in SpotOptim's convention
  `fun(X)` with `X` of shape `(n, d)`), the vectorization is never exploited
  in parallel mode

When the objective is cheap relative to process overhead (e.g. a fast
simulation that runs in < 1 second), single-point dispatch is inefficient.

**Change:** Accumulate multiple candidate points from completed `search` tasks
and dispatch them as a single batch `eval` task when enough candidates are
available:

```python
BATCH_SIZE = n_jobs  # or a new `eval_batch_size` parameter

pending_cands = []   # candidates awaiting evaluation

# When a search task completes:
pending_cands.append(x_cand)

if len(pending_cands) >= BATCH_SIZE or no_active_searches():
    X_batch = np.vstack(pending_cands)
    pending_cands.clear()
    fut = eval_pool.submit(remote_batch_eval_wrapper, dill.dumps((self, X_batch)))
```

The batch eval wrapper unpacks and calls `fun(X_batch)` once, returning
`(X_batch, y_batch)`.

**Expected benefit:**
- Amortizes process-spawn + IPC overhead across `BATCH_SIZE` evaluations
- Allows user functions that are naturally vectorized to exploit that property
- Reduces total number of `dill.dumps` calls by factor `BATCH_SIZE`

**New parameter:** `eval_batch_size: int = 1` (default preserves current
behaviour; set to `n_jobs` or higher to activate batching).

**Compatibility:** Default `eval_batch_size=1` is fully backward compatible.
Users with vectorized objective functions explicitly opt in.

**Risk:** Low.  Batch dispatch slightly increases the latency before the first
result of a batch is available, which can temporarily reduce steady-state
throughput when `fun` is very slow.  The configurable `eval_batch_size`
parameter lets users tune the trade-off.

---

### Release 0.10.0 — Improvement D: Free-Threaded (No-GIL) Awareness

**Background:** Python 3.13 introduced experimental free-threaded builds
(`python3.13t`, compile flag `--disable-gil`).  When the GIL is disabled,
`ThreadPoolExecutor` achieves true CPU-level parallelism for any Python code,
not just GIL-releasing extensions.  By release 0.9.0, free-threaded Python
is expected to be more mature and the default in some environments.

**Dependency:** This improvement extends Improvement C (ThreadPoolExecutor for
search tasks).  With the GIL disabled, the `ThreadPoolExecutor` that already
runs search tasks in 0.7.0 automatically gains full multi-core parallelism
for any Python-level computation inside `suggest_next_infill_point()` — no
code change is required for that path.

**Change:** Add GIL-status detection to the executor selection logic:

```python
import sys

def _is_gil_disabled() -> bool:
    """Return True if running on a free-threaded Python build."""
    return not getattr(sys, "_is_gil_enabled", lambda: True)()

def _build_executors(n_jobs: int):
    if _is_gil_disabled():
        # Free-threaded: threads give true CPU parallelism; no process pool needed.
        # Use threads for both search and eval (eval isolation provided by the GIL-free runtime).
        search_pool = ThreadPoolExecutor(max_workers=n_jobs)
        eval_pool   = ThreadPoolExecutor(max_workers=n_jobs)
    else:
        # Standard GIL build: keep hybrid design from 0.7.0.
        search_pool = ThreadPoolExecutor(max_workers=n_jobs)
        eval_pool   = ProcessPoolExecutor(max_workers=n_jobs)
    return eval_pool, search_pool
```

When the GIL is disabled and both pools are `ThreadPoolExecutor`, the `dill`
serialization path for eval tasks can also be eliminated — `fun` is accessible
directly from the shared heap.

**Additional documentation:** Update the user guide and docstring for `n_jobs`
to note the recommended Python version and build type for maximum parallelism.

**Compatibility:** The detection is purely additive.  On standard GIL-enabled
Python (the vast majority of deployments through at least 2026), behaviour is
identical to 0.8.0.

**Risk:** Low.  The free-threaded build is opt-in at the Python level.
spotoptim does not force users onto it; it merely exploits it when present.

---

## 6. Summary

| Release | Improvement | Key Change | Risk | Status |
|---------|-------------|------------|------|--------|
| **0.7.0** | E — `n_jobs=-1` | Resolve `-1` to `os.cpu_count()` | Minimal | ✅ Done |
| **0.8.0** | C — Threads for search | `ThreadPoolExecutor` for `suggest_next_infill_point`; `ProcessPoolExecutor` kept for `eval`; `threading.Lock` guards surrogate | Low | ✅ Done |
| **0.9.0** | F — Batch eval | Accumulate candidates; one `fun(X_batch)` call per batch | Low | Planned |
| **0.10.0** | D — No-GIL | Detect `sys._is_gil_enabled()`; use threads for eval too | Low | Planned |

Improvements A (shared memory) and B (SciPy `workers=`) are **not planned**
because A is superseded by C and B conflicts with the existing `vectorized=True`
optimisation.
