"""Shared pytest configuration for the spotoptim test suite.

Slow-test registry
------------------
The suite contains ~1.7k tests; a few dozen end-to-end optimization runs
dominate the wall-clock. Those are tagged ``@pytest.mark.slow`` here, in one
central place, so that:

* the fast path (PR gate, pre-push hook) runs ``pytest -m "not slow"`` and
  skips them, and
* the nightly / full CI job runs everything.

Entries in :data:`SLOW_NODEIDS` are matched against each test's node id with
the parametrization suffix (``[...]``) stripped. An entry may be:

* a file            ``tests/test_x0_starting_point.py``
* a class           ``tests/test_batch_eval.py::TestBatchEvalEndToEnd``
* a single test     ``tests/test_save_load.py::TestEdgeCases::test_reproducibility_after_load``

This list was derived from ``pytest --durations`` (tests taking >~8 s).
To refresh it, run ``uv run pytest tests/ -n auto --durations=50`` and update
the entries below. Stale entries that no longer match anything are harmless.
New slow tests may instead be decorated directly with ``@pytest.mark.slow``.
"""

import os

# Pin native thread pools to 1 thread per worker BEFORE numpy/scipy/torch are
# imported. pytest-xdist runs one worker per core; without this, each worker's
# BLAS/OpenMP pool spawns more threads and they thrash — on CI this made small
# linear-algebra tests run ~50-80x slower. setdefault keeps any value the
# environment already set (e.g. the CI workflow exports the same vars).
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import pytest  # noqa: E402  — must come after the thread-var setup above

# Heaviest tests (>~8 s each), grouped by file. Class/file entries also cover
# their faster sibling integration tests, which belong in the full suite too.
SLOW_NODEIDS = {
    # --- whole files (every test is a full optimization run) ---
    "tests/test_x0_starting_point.py",
    # --- integration test classes ---
    "tests/test_batch_eval.py::TestBatchEvalEndToEnd",
    "tests/test_termination_criteria.py::TestTerminationCriteria",
    "tests/test_thread_pool_search.py::TestHybridExecutorEndToEnd",
    "tests/test_no_gil_awareness.py::TestSimulatedNoGilPath",
    "tests/test_no_gil_awareness.py::TestGilBuildEndToEnd",
    "tests/test_multiobjective.py::TestMultiObjectiveOptimization",
    "tests/test_transformations.py::TestTransformationOptimization",
    "tests/test_reproducibility_comprehensive.py::TestSpotOptimReproducibility",
    "tests/test_optimize_refactored_methods.py::TestOptimizeIntegration",
    # --- networked integration test: skip from the fast gate (the request
    #     timeout in remote.py bounds it; the nightly full run still exercises it) ---
    "tests/test_objective_remote.py::test_objective_remote",
    # --- individual heavy tests ---
    "tests/test_early_stopping.py::test_max_restarts_does_not_trigger_with_improvement",
    "tests/test_parallel_optimization.py::TestParallelOptimization::test_parallel_execution_basic",
    "tests/test_multiobjective.py::TestMultiObjectiveEdgeCases::test_many_objectives",
    "tests/test_initial_design_nan_handling.py::test_initial_design_with_mixed_nan_inf",
    "tests/test_cookbook_examples.py::test_example_4_nelder_mead",
    "tests/test_spotoptim_mlp_surrogate.py::test_mlp_surrogate_uncertainty_in_loop",
    "tests/test_save_load.py::TestEdgeCases::test_reproducibility_after_load",
    "tests/test_run_sequential_loop_example.py::test_run_sequential_loop_example",
    "tests/test_factor_variables.py::TestFactorVariables::test_many_factor_levels",
    "tests/test_spotoptim_deep.py::TestConvergenceQuality::test_sphere_2d_converges_near_origin",
    "tests/test_spotoptim_deep.py::TestSeedReproducibility::test_same_seed_same_results",
    "tests/test_max_iter_validation.py::TestMaxIterValidation::test_max_iter_greater_than_n_initial_works",
    "tests/test_termination.py::TestBackwardCompatibility::test_old_behavior_without_max_time",
    "tests/test_termination.py::TestMaxIterTermination::test_max_iter_with_custom_initial_design",
    "tests/test_tolerance_x.py::TestToleranceXFloatVariables::test_no_duplicate_evaluations_float",
    "tests/test_tolerance_x.py::TestToleranceXFactorVariables::test_no_duplicate_evaluations_factors",
    "tests/test_spotoptim.py::TestSpotOptimOptimize::test_optimize_with_seed_reproducibility",
    "tests/test_parallel_reporting.py::test_parallel_reporting",
    "tests/test_parallel_merging.py::test_parallel_merging",
    "tests/test_deterministic.py::TestDeterministicBehavior::test_deterministic_with_provided_initial_design",
    "tests/test_acquisition_failure.py::TestAcquisitionFailureWithVariableTypes::test_acquisition_failure_with_mixed_variables",
    "tests/test_spot_optim_args.py::test_kwargs_only",
    "tests/test_tensorboard.py::TestTensorBoardIntegration::test_tensorboard_with_custom_var_names",
    "tests/test_tensorboard_clean.py::TestTensorBoardClean::test_clean_with_optimization_run",
}


def _is_slow(nodeid: str) -> bool:
    """Return True if ``nodeid`` is covered by a SLOW_NODEIDS entry."""
    base = nodeid.split("[", 1)[0]  # drop parametrization suffix
    return any(base == entry or base.startswith(entry + "::") for entry in SLOW_NODEIDS)


def pytest_collection_modifyitems(config, items):
    """Tag registered heavy tests with the ``slow`` marker during collection.

    Runs before ``-m`` deselection, so ``pytest -m "not slow"`` skips them.
    """
    slow = pytest.mark.slow
    for item in items:
        if _is_slow(item.nodeid):
            item.add_marker(slow)
