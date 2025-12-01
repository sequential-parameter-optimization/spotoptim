"""Tests for TensorBoard logging functionality."""

import numpy as np
import pytest
import os
import shutil
from spotoptim import SpotOptim


class TestTensorBoardParameters:
    """Test TensorBoard parameter initialization."""

    def test_tensorboard_log_default_false(self):
        """Test that tensorboard_log defaults to False."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
        )
        assert opt.tensorboard_log is False
        assert opt.tb_writer is None

    def test_tensorboard_log_enabled(self):
        """Test enabling TensorBoard logging."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            tensorboard_log=True,
        )
        assert opt.tensorboard_log is True
        assert opt.tb_writer is not None

        # Cleanup
        opt._close_tensorboard_writer()
        if os.path.exists(opt.tensorboard_path):
            shutil.rmtree(opt.tensorboard_path)

    def test_tensorboard_custom_path(self):
        """Test custom TensorBoard path."""
        custom_path = "test_runs/custom_test"
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            tensorboard_log=True,
            tensorboard_path=custom_path,
        )
        assert opt.tensorboard_path == custom_path
        assert os.path.exists(custom_path)

        # Cleanup
        opt._close_tensorboard_writer()
        if os.path.exists("test_runs"):
            shutil.rmtree("test_runs")

    def test_tensorboard_default_path_creation(self):
        """Test that default path is created with timestamp."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            tensorboard_log=True,
        )
        assert opt.tensorboard_path is not None
        assert opt.tensorboard_path.startswith("runs/spotoptim_")
        assert os.path.exists(opt.tensorboard_path)

        # Cleanup
        opt._close_tensorboard_writer()
        if os.path.exists(opt.tensorboard_path):
            shutil.rmtree(opt.tensorboard_path)


class TestTensorBoardLogging:
    """Test TensorBoard logging during optimization."""

    def test_logging_creates_files(self):
        """Test that logging creates event files."""
        np.random.seed(42)
        tensorboard_path = "test_runs/logging_test"

        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=15,
            n_initial=10,
            tensorboard_log=True,
            tensorboard_path=tensorboard_path,
            seed=42,
            verbose=False,
        )

        result = opt.optimize()

        # Check that TensorBoard directory exists
        assert os.path.exists(tensorboard_path)

        # Check that event files were created
        files = os.listdir(tensorboard_path)
        event_files = [f for f in files if f.startswith("events.out.tfevents")]
        assert len(event_files) > 0, "No TensorBoard event files created"

        # Cleanup
        if os.path.exists("test_runs"):
            shutil.rmtree("test_runs")

    def test_logging_with_noisy_function(self):
        """Test logging with noisy optimization."""
        np.random.seed(123)
        tensorboard_path = "test_runs/noisy_logging"

        def noisy_sphere(X):
            base = np.sum(X**2, axis=1)
            noise = np.random.normal(0, 0.1, size=base.shape)
            return base + noise

        opt = SpotOptim(
            fun=noisy_sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=20,
            n_initial=10,
            repeats_initial=2,
            tensorboard_log=True,
            tensorboard_path=tensorboard_path,
            seed=123,
            verbose=False,
        )

        result = opt.optimize()

        # Check that logging completed without errors
        assert result.success is True
        assert os.path.exists(tensorboard_path)

        # Cleanup
        if os.path.exists("test_runs"):
            shutil.rmtree("test_runs")

    def test_logging_disabled_by_default(self):
        """Test that no logging occurs when disabled."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=15,
            n_initial=10,
            tensorboard_log=False,  # Explicitly disabled
            seed=42,
            verbose=False,
        )

        result = opt.optimize()

        # tb_writer should be None
        assert opt.tb_writer is None

        # No default runs directory should be created
        # (unless specifically set)
        assert result.success is True

    def test_writer_closed_after_optimization(self):
        """Test that writer is properly closed after optimization."""
        tensorboard_path = "test_runs/close_test"

        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5)],
            max_iter=10,
            n_initial=5,
            tensorboard_log=True,
            tensorboard_path=tensorboard_path,
            seed=42,
            verbose=False,
        )

        # Writer should exist before optimization
        assert opt.tb_writer is not None

        result = opt.optimize()

        # Writer should be closed after optimization
        assert not hasattr(opt, "tb_writer") or opt.tb_writer is None

        # Cleanup
        if os.path.exists("test_runs"):
            shutil.rmtree("test_runs")


class TestTensorBoardMethods:
    """Test individual TensorBoard methods."""

    def test_init_tensorboard_writer(self):
        """Test _init_tensorboard_writer method."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5)],
            max_iter=5,
            n_initial=3,
            tensorboard_log=True,
            verbose=False,
        )

        assert opt.tb_writer is not None
        assert opt.tensorboard_path is not None

        # Cleanup
        opt._close_tensorboard_writer()
        if os.path.exists(opt.tensorboard_path):
            shutil.rmtree(opt.tensorboard_path)

    def test_write_tensorboard_scalars(self):
        """Test _write_tensorboard_scalars method."""
        np.random.seed(55)
        tensorboard_path = "test_runs/scalars_test"

        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            tensorboard_log=True,
            tensorboard_path=tensorboard_path,
            seed=55,
            verbose=False,
        )

        # Initialize some data
        opt.X_ = np.array([[1, 2], [3, 4]])
        opt.y_ = np.array([5.0, 25.0])
        opt.update_stats()

        # Should not raise an error
        opt._write_tensorboard_scalars()

        # Cleanup
        opt._close_tensorboard_writer()
        if os.path.exists("test_runs"):
            shutil.rmtree("test_runs")

    def test_write_tensorboard_hparams(self):
        """Test _write_tensorboard_hparams method."""
        tensorboard_path = "test_runs/hparams_test"

        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            tensorboard_log=True,
            tensorboard_path=tensorboard_path,
            verbose=False,
        )

        # Test writing hparams
        X_test = np.array([1.5, 2.5])
        y_test = 10.0

        # Should not raise an error
        opt._write_tensorboard_hparams(X_test, y_test)

        # Cleanup
        opt._close_tensorboard_writer()
        if os.path.exists("test_runs"):
            shutil.rmtree("test_runs")

    def test_close_tensorboard_writer(self):
        """Test _close_tensorboard_writer method."""
        tensorboard_path = "test_runs/close_method_test"

        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5)],
            max_iter=5,
            n_initial=3,
            tensorboard_log=True,
            tensorboard_path=tensorboard_path,
            verbose=False,
        )

        assert opt.tb_writer is not None

        opt._close_tensorboard_writer()

        assert not hasattr(opt, "tb_writer") or opt.tb_writer is None

        # Cleanup
        if os.path.exists("test_runs"):
            shutil.rmtree("test_runs")


class TestTensorBoardIntegration:
    """Integration tests for TensorBoard with other features."""

    def test_tensorboard_with_ocba(self):
        """Test TensorBoard logging with OCBA."""
        np.random.seed(99)
        tensorboard_path = "test_runs/ocba_integration"

        def noisy_func(X):
            return np.sum(X**2, axis=1) + np.random.normal(0, 0.1, X.shape[0])

        opt = SpotOptim(
            fun=noisy_func,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=30,
            n_initial=10,
            repeats_initial=2,
            ocba_delta=2,
            tensorboard_log=True,
            tensorboard_path=tensorboard_path,
            seed=99,
            verbose=False,
        )

        result = opt.optimize()

        assert result.success is True
        assert os.path.exists(tensorboard_path)

        # Cleanup
        if os.path.exists("test_runs"):
            shutil.rmtree("test_runs")

    def test_tensorboard_with_dimension_reduction(self):
        """Test TensorBoard with fixed dimensions."""
        tensorboard_path = "test_runs/dim_reduction"

        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (2, 2), (-5, 5)],  # Middle dimension fixed
            max_iter=15,
            n_initial=8,
            tensorboard_log=True,
            tensorboard_path=tensorboard_path,
            seed=42,
            verbose=False,
        )

        result = opt.optimize()

        assert result.success is True
        assert os.path.exists(tensorboard_path)

        # Cleanup
        if os.path.exists("test_runs"):
            shutil.rmtree("test_runs")

    def test_tensorboard_with_custom_var_names(self):
        """Test TensorBoard with custom variable names."""
        tensorboard_path = "test_runs/var_names"

        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            var_name=["alpha", "beta"],
            max_iter=15,
            n_initial=8,
            tensorboard_log=True,
            tensorboard_path=tensorboard_path,
            seed=42,
            verbose=False,
        )

        result = opt.optimize()

        assert result.success is True
        assert os.path.exists(tensorboard_path)

        # Cleanup
        if os.path.exists("test_runs"):
            shutil.rmtree("test_runs")


class TestTensorBoardEdgeCases:
    """Test edge cases for TensorBoard logging."""

    def test_tensorboard_with_max_iter_equals_n_initial(self):
        """Test logging when no sequential iterations occur."""
        tensorboard_path = "test_runs/no_iterations"

        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5)],
            max_iter=5,
            n_initial=5,  # Same as max_iter
            tensorboard_log=True,
            tensorboard_path=tensorboard_path,
            seed=42,
            verbose=False,
        )

        result = opt.optimize()

        assert result.success is True
        assert result.nit == 0  # No iterations
        assert os.path.exists(tensorboard_path)

        # Cleanup
        if os.path.exists("test_runs"):
            shutil.rmtree("test_runs")

    def test_tensorboard_with_very_few_iterations(self):
        """Test TensorBoard with minimal optimization."""
        tensorboard_path = "test_runs/minimal_test"

        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5)],
            max_iter=6,
            n_initial=5,
            tensorboard_log=True,
            tensorboard_path=tensorboard_path,
            seed=42,
            verbose=False,
        )

        result = opt.optimize()

        # Should complete successfully with just 1 iteration
        assert result.success is True
        assert result.nit == 1
        assert os.path.exists(tensorboard_path)

        # Cleanup
        if os.path.exists("test_runs"):
            shutil.rmtree("test_runs")
