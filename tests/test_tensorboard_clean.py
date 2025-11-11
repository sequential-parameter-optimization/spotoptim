"""
Tests for TensorBoard log cleaning functionality.
"""

import pytest
import numpy as np
import os
import shutil
from spotoptim import SpotOptim


class TestTensorBoardClean:
    """Test TensorBoard log directory cleaning."""

    def setup_method(self):
        """Create a temporary runs directory with some test subdirectories."""
        # Ensure runs directory exists
        os.makedirs("runs", exist_ok=True)
        
        # Create some test subdirectories
        test_dirs = [
            "runs/test_log_1",
            "runs/test_log_2",
            "runs/spotoptim_20231101_120000",
        ]
        for d in test_dirs:
            os.makedirs(d, exist_ok=True)
            # Create a dummy file in each
            with open(os.path.join(d, "dummy.txt"), "w") as f:
                f.write("test")

    def teardown_method(self):
        """Clean up runs directory after tests."""
        if os.path.exists("runs"):
            shutil.rmtree("runs")

    def test_tensorboard_clean_default_false(self):
        """Test that tensorboard_clean defaults to False."""
        optimizer = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
        )
        assert optimizer.tensorboard_clean is False

    def test_tensorboard_clean_enabled(self):
        """Test that tensorboard_clean can be enabled."""
        optimizer = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            tensorboard_clean=True,
        )
        assert optimizer.tensorboard_clean is True

    def test_clean_removes_old_logs(self):
        """Test that enabling clean removes old TensorBoard logs."""
        # Count initial subdirectories
        initial_count = len([
            d for d in os.listdir("runs")
            if os.path.isdir(os.path.join("runs", d))
        ])
        assert initial_count > 0, "Test setup should create subdirectories"

        # Create optimizer with clean enabled (but no logging)
        optimizer = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            tensorboard_clean=True,
            verbose=False,
        )

        # Check that runs directory is now empty or doesn't exist
        if os.path.exists("runs"):
            remaining = [
                d for d in os.listdir("runs")
                if os.path.isdir(os.path.join("runs", d))
            ]
            assert len(remaining) == 0, "All subdirectories should be removed"

    def test_clean_with_logging(self):
        """Test that clean works together with logging enabled."""
        # Create optimizer with both clean and logging enabled
        optimizer = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            tensorboard_log=True,
            tensorboard_clean=True,
            verbose=False,
        )

        # Old logs should be cleaned, but new log directory should exist
        subdirs = [
            d for d in os.listdir("runs")
            if os.path.isdir(os.path.join("runs", d))
        ]
        
        # Should have exactly one directory (the new one)
        assert len(subdirs) == 1
        # Should start with 'spotoptim_'
        assert subdirs[0].startswith("spotoptim_")

    def test_clean_when_runs_does_not_exist(self):
        """Test that clean handles missing runs directory gracefully."""
        # Remove runs directory
        if os.path.exists("runs"):
            shutil.rmtree("runs")

        # Should not raise an error
        optimizer = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            tensorboard_clean=True,
            verbose=False,
        )
        
        # Optimizer should be created successfully
        assert optimizer.tensorboard_clean is True

    def test_clean_with_verbose(self, capsys):
        """Test that clean provides verbose output when enabled."""
        optimizer = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            tensorboard_clean=True,
            verbose=True,
        )

        captured = capsys.readouterr()
        # Should mention cleaning
        assert "Cleaned" in captured.out or "Removed" in captured.out

    def test_clean_does_not_affect_files(self):
        """Test that clean only removes directories, not files."""
        # Create a file in runs directory
        test_file = "runs/test_file.txt"
        with open(test_file, "w") as f:
            f.write("test")

        optimizer = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            tensorboard_clean=True,
            verbose=False,
        )

        # File should still exist
        assert os.path.exists(test_file)

    def test_clean_disabled_preserves_logs(self):
        """Test that logs are preserved when clean is disabled."""
        # Count initial subdirectories
        initial_dirs = [
            d for d in os.listdir("runs")
            if os.path.isdir(os.path.join("runs", d))
        ]
        initial_count = len(initial_dirs)
        assert initial_count > 0

        # Create optimizer without clean
        optimizer = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            tensorboard_clean=False,
            verbose=False,
        )

        # All original directories should still exist
        remaining_dirs = [
            d for d in os.listdir("runs")
            if os.path.isdir(os.path.join("runs", d))
        ]
        assert len(remaining_dirs) == initial_count

    def test_clean_with_optimization_run(self):
        """Test clean followed by a full optimization run with logging."""
        # Run optimization with both clean and logging
        optimizer = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=15,
            n_initial=8,
            tensorboard_log=True,
            tensorboard_clean=True,
            seed=42,
            verbose=False,
        )
        
        result = optimizer.optimize()
        
        # Should have successful optimization
        assert result.success
        
        # Should have exactly one log directory
        subdirs = [
            d for d in os.listdir("runs")
            if os.path.isdir(os.path.join("runs", d))
        ]
        assert len(subdirs) == 1
        
        # Log directory should contain TensorBoard event files
        log_dir = os.path.join("runs", subdirs[0])
        files = os.listdir(log_dir)
        assert any("events.out.tfevents" in f for f in files)


class TestTensorBoardCleanEdgeCases:
    """Test edge cases for TensorBoard cleaning."""

    def teardown_method(self):
        """Clean up runs directory after tests."""
        if os.path.exists("runs"):
            shutil.rmtree("runs")

    def test_clean_with_custom_tensorboard_path(self):
        """Test that clean doesn't interfere with custom paths."""
        # Create custom directory
        custom_path = "my_logs/experiment_1"
        os.makedirs(custom_path, exist_ok=True)
        
        # Create some old logs in runs
        os.makedirs("runs/old_log", exist_ok=True)
        
        try:
            # Create optimizer with clean and custom path
            optimizer = SpotOptim(
                fun=lambda X: np.sum(X**2, axis=1),
                bounds=[(-5, 5), (-5, 5)],
                max_iter=10,
                n_initial=5,
                tensorboard_log=True,
                tensorboard_path=custom_path,
                tensorboard_clean=True,
                verbose=False,
            )
            
            # Runs directory should be cleaned
            if os.path.exists("runs"):
                subdirs = [
                    d for d in os.listdir("runs")
                    if os.path.isdir(os.path.join("runs", d))
                ]
                assert len(subdirs) == 0
            
            # Custom directory should still exist
            assert os.path.exists(custom_path)
            
        finally:
            if os.path.exists("my_logs"):
                shutil.rmtree("my_logs")

    def test_clean_with_nested_directories(self):
        """Test that clean handles nested structures correctly."""
        # Create nested structure
        os.makedirs("runs/parent/child", exist_ok=True)
        
        optimizer = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            tensorboard_clean=True,
            verbose=False,
        )
        
        # Should remove parent directory (which contains child)
        if os.path.exists("runs"):
            subdirs = [
                d for d in os.listdir("runs")
                if os.path.isdir(os.path.join("runs", d))
            ]
            assert len(subdirs) == 0

    def test_clean_multiple_times(self):
        """Test that clean can be called multiple times safely."""
        os.makedirs("runs/log1", exist_ok=True)
        
        # First clean
        opt1 = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            tensorboard_clean=True,
            verbose=False,
        )
        
        # Recreate some logs
        os.makedirs("runs/log2", exist_ok=True)
        
        # Second clean
        opt2 = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            tensorboard_clean=True,
            verbose=False,
        )
        
        # Should work without errors
        assert opt2.tensorboard_clean is True
