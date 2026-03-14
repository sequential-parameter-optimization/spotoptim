# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
from spotoptim import SpotOptim


def test_max_iter_budget_with_restarts():
    """Verify that max_iter is treated as a global budget across restarts."""

    class CounterObj:
        def __init__(self):
            self.count = 0

        def __call__(self, X):
            # Return same value to force stagnation and restarts
            n_points = X.shape[0]
            self.count += n_points
            return np.ones(n_points) * 10.0

    obj = CounterObj()

    max_iter = 15
    n_initial = 5

    # Configure so it restarts quickly
    # restart_after_n=2 means if 2 iterations have 0 success rate, restart.
    # With constant function, success rate will be 0 always.
    optimizer = SpotOptim(
        fun=obj,
        bounds=[(-5, 5)],
        max_iter=max_iter,
        n_initial=n_initial,
        restart_after_n=2,
        max_time=0.1,  # Stop after 6 seconds to prevent infinite loop if max_iter is broken
        verbose=True,
    )

    optimizer.optimize()

    print(f"Total evaluations: {obj.count}")
    print(f"Restarts results: {len(optimizer.restarts_results_)}")

    # If budget is global, count should be exactly max_iter (or slightly less if it stopped early, but here it should hit limit)
    # Actually, if the last restart doesn't finish initial design because of check?
    # Current implementation doesn't check global, so it will likely exceed.

    assert (
        obj.count <= max_iter
    ), f"Total evaluations {obj.count} exceeded max_iter {max_iter}"

    # We expect at least one restart if logic works as claimed currently (it restarts and resets budget)
    # With global budget:
    # Run 1: 5 initial + 10 iterations = 15.
    # Wait, max_iter=15. So 5 initial + 10 opt steps.
    # Stagnation happens. restart_after_n=2.
    # Iter 1: success=0.
    # Iter 2: success=0.
    # Restart triggered.
    # Run 1 used: 5 + 2 = 7 evals.
    # Remaining budget: 15 - 7 = 8.
    # Run 2: 5 initial. Remaining 3.
    # Iter 1, Iter 2. Restart.
    # Run 2 used: 5 + 2 = 7.
    # Remaining: 15 - 14 = 1.
    # Run 3: Needs 5 initial. Cannot start.
    # Total should be 14 or 15.

    # Current buggy behavior:
    # Run 1: 7 evals.
    # Run 2: 7 evals (because it thinks it has 15 budget)
    # Run 3: 7 evals.
    # ... loops until 60 minutes or indefinite?
    # Wait, optimize loop currently breaks if status != RESTART or if restarts_results_ check?
    # No, it loops `while True`. If `_optimize_single_run` terminates with "FINISHED" because `max_iter` reached (per run), it breaks.
    # But here `_optimize_single_run` returns "RESTART".
    # So it will loop forever! Or until `max_time`.
    # This is bad. It confirms the bug is severe.

    pass


if __name__ == "__main__":
    test_max_iter_budget_with_restarts()
