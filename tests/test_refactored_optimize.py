# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from spotoptim.SpotOptim import SpotOptim
from scipy.optimize import OptimizeResult
import time

class TestRefactoredOptimize:
    """Test suite for the refactored SpotOptim.optimize structure."""

    @pytest.fixture
    def spot_optim(self):
        """Fixture to provide a SpotOptim instance."""
        def sphere(X):
            return np.sum(X**2, axis=1)
        
        return SpotOptim(
            fun=sphere,
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            max_iter=10,
            n_jobs=1,
            seed=42,
            verbose=False
        )

    def test_optimize_structure_sequential(self, spot_optim):
        """Test that optimize calls _execute_optimization_run and loop works for sequential."""
        spot_optim.n_jobs = 1
        
        # Mock _execute_optimization_run to return a finished result immediately
        mock_result = OptimizeResult(
            x=np.zeros(2), fun=0.0, nfev=10, nit=5, success=True, message="Finished", X=np.zeros((10,2)), y=np.zeros(10)
        )
        spot_optim._execute_optimization_run = MagicMock(return_value=("FINISHED", mock_result))

        result = spot_optim.optimize()

        assert result == mock_result
        spot_optim._execute_optimization_run.assert_called_once()
        # Verify arguments passed (checking structure)
        call_args = spot_optim._execute_optimization_run.call_args
        assert call_args[1]['y0_known'] is None

    def test_optimize_structure_restart(self, spot_optim):
        """Test restart logic in optimize loop."""
        spot_optim.n_jobs = 1
        spot_optim.max_iter = 20
        spot_optim.n_initial = 5
        spot_optim.restart_inject_best = True
        
        # Mock results
        res1 = OptimizeResult(fun=1.0, x=np.array([1.0, 1.0]), nfev=10, nit=5, X=np.ones((10, 2)), y=np.ones(10))
        res2 = OptimizeResult(fun=0.1, x=np.array([0.1, 0.1]), nfev=10, nit=5, X=np.ones((10, 2)), y=np.ones(10)*0.1)
        
        # Side effect: First call returns RESTART, second returns FINISHED
        spot_optim._execute_optimization_run = MagicMock(side_effect=[
            ("RESTART", res1),
            ("FINISHED", res2)
        ])
        
        # Mock _validate_x0 to just return input (simplification)
        spot_optim._validate_x0 = MagicMock(side_effect=lambda x: x)

        result = spot_optim.optimize()

        assert spot_optim._execute_optimization_run.call_count == 2
        assert result.fun == 0.1
        assert len(spot_optim.restarts_results_) == 2
        
        # Check if second call received the best value from first run
        args2 = spot_optim._execute_optimization_run.call_args_list[1]
        assert args2[1]['y0_known'] == 1.0

    def test_dispatch_sequential(self, spot_optim):
        """Verify dispatch calls _optimize_sequential_run when n_jobs=1."""
        spot_optim.n_jobs = 1
        spot_optim._optimize_sequential_run = MagicMock(return_value=("FINISHED", MagicMock()))
        spot_optim._optimize_steady_state = MagicMock()

        spot_optim._execute_optimization_run(timeout_start=time.time())
        
        spot_optim._optimize_sequential_run.assert_called_once()
        spot_optim._optimize_steady_state.assert_not_called()

    def test_dispatch_parallel(self, spot_optim):
        """Verify dispatch calls _optimize_steady_state when n_jobs>1."""
        spot_optim.n_jobs = 2
        spot_optim._optimize_sequential_run = MagicMock()
        spot_optim._optimize_steady_state = MagicMock(return_value=("FINISHED", MagicMock()))

        spot_optim._execute_optimization_run(timeout_start=time.time())
        
        spot_optim._optimize_steady_state.assert_called_once()
        spot_optim._optimize_sequential_run.assert_not_called()

    def test_initialize_run_calls(self, spot_optim):
        """Verify _initialize_run calls necessary setup methods."""
        spot_optim._set_seed = MagicMock()
        spot_optim.get_initial_design = MagicMock(return_value=np.zeros((5, 2)))
        spot_optim._curate_initial_design = MagicMock(return_value=np.zeros((5, 2)))
        spot_optim._evaluate_function = MagicMock(return_value=np.zeros(5))
        spot_optim._init_tensorboard = MagicMock()

        X0, y0 = spot_optim._initialize_run(X0=None, y0_known=None)
        
        spot_optim._set_seed.assert_called_once()
        spot_optim.get_initial_design.assert_called_once()
        spot_optim._evaluate_function.assert_called_once()
        # _init_tensorboard should NOT be called here anymore
        spot_optim._init_tensorboard.assert_not_called()

    def test_optimize_sequential_run_calls_init_tensorboard(self, spot_optim):
        """Verify _optimize_sequential_run calls _init_tensorboard."""
        spot_optim._initialize_run = MagicMock(return_value=(np.zeros((5, 2)), np.zeros(5)))
        spot_optim._rm_NA_values = MagicMock(return_value=(np.zeros((5, 2)), np.zeros(5), 5))
        spot_optim._check_size_initial_design = MagicMock()
        spot_optim._init_storage = MagicMock()
        spot_optim.update_stats = MagicMock()
        spot_optim._get_best_xy_initial_design = MagicMock()
        spot_optim._run_sequential_loop = MagicMock(return_value=("FINISHED", MagicMock()))
        spot_optim._init_tensorboard = MagicMock()

        spot_optim._optimize_sequential_run(timeout_start=time.time())

        spot_optim._init_tensorboard.assert_called_once()

    def test_run_sequential_loop_termination_iter(self, spot_optim):
        """Verify loop terminates on max evaluations."""
        spot_optim.max_iter = 5
        spot_optim.max_time = 100
        spot_optim.y_ = np.zeros(5) # Already at max
        spot_optim.best_y_ = 0.0
        spot_optim.best_x_ = np.zeros(2)
        
        status, res = spot_optim._run_sequential_loop(timeout_start=time.time(), effective_max_iter=5)
        
        assert status == "FINISHED"
        assert res.success is True
        assert "maximum evaluations" in res.message

    def test_optimize_integration_sequential(self):
        """Integration test for full sequential run."""
        def sphere(X):
            X = np.atleast_2d(X)
            return np.sum(X**2, axis=1)

        opt = SpotOptim(
            fun=sphere,
            bounds=[(-5, 5)],
            n_initial=3,
            max_iter=6,
            seed=42
        )
        
        # Should run without error
        res = opt.optimize()
        assert res.success
        assert len(opt.y_) >= 6
