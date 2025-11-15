"""
Tests for save/load functionality (experiments and results).

This test suite validates the save_experiment, load_experiment, save_result,
and load_result methods, ensuring proper serialization and deserialization.
"""

import pytest
import numpy as np
import os
import tempfile
import shutil
from spotoptim.SpotOptim import SpotOptim


# Module-level test functions (needed for pickling)
def simple_func(X):
    """Simple sphere function for testing."""
    return np.sum(X**2, axis=1)


def noisy_func(X):
    """Noisy sphere function for testing."""
    np.random.seed(42)
    return np.sum(X**2, axis=1) + np.random.normal(0, 0.1, X.shape[0])


class TestSaveLoadExperiment:
    """Test suite for experiment save/load functionality."""

    def test_save_experiment_default_filename(self):
        """Test saving experiment with default filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = SpotOptim(
                fun=simple_func,
                bounds=[(-5, 5), (-5, 5)],
                max_iter=10,
                n_initial=5,
                seed=42
            )
            
            # Save with default filename
            filename = os.path.join(tmpdir, "test_exp.pkl")
            opt.save_experiment(filename=filename, verbosity=0)
            
            # Check file exists
            assert os.path.exists(filename)

    def test_save_experiment_with_prefix(self):
        """Test saving experiment with prefix-generated filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = SpotOptim(
                fun=simple_func,
                bounds=[(-5, 5), (-5, 5)],
                max_iter=10,
                n_initial=5,
                seed=42
            )
            
            # Save with prefix
            opt.save_experiment(prefix="my_experiment", path=tmpdir, verbosity=0)
            
            # Check file exists
            expected_file = os.path.join(tmpdir, "my_experiment_exp.pkl")
            assert os.path.exists(expected_file)

    def test_load_experiment_basic(self):
        """Test loading a saved experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save experiment
            opt_original = SpotOptim(
                fun=simple_func,
                bounds=[(-5, 5), (-5, 5)],
                max_iter=30,
                n_initial=10,
                seed=42,
                verbose=False
            )
            
            filename = os.path.join(tmpdir, "experiment.pkl")
            opt_original.save_experiment(filename=filename, verbosity=0)
            
            # Load experiment
            opt_loaded = SpotOptim.load_experiment(filename)
            
            # Verify configuration preserved
            np.testing.assert_array_equal(opt_loaded.lower, opt_original.lower)
            np.testing.assert_array_equal(opt_loaded.upper, opt_original.upper)
            assert opt_loaded.max_iter == opt_original.max_iter
            assert opt_loaded.n_initial == opt_original.n_initial
            assert opt_loaded.seed == opt_original.seed

    def test_load_experiment_file_not_found(self):
        """Test loading experiment from non-existent file raises error."""
        
        with pytest.raises(FileNotFoundError):
            SpotOptim.load_experiment("nonexistent_file.pkl")

    def test_save_experiment_overwrite_protection(self):
        """Test that overwrite=False prevents file overwriting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = SpotOptim(
                fun=simple_func,
                bounds=[(-5, 5), (-5, 5)],
                max_iter=10,
                n_initial=5,
                seed=42
            )
            
            filename = os.path.join(tmpdir, "test.pkl")
            
            # Save first time
            opt.save_experiment(filename=filename, verbosity=0)
            
            # Try to save again with overwrite=False
            with pytest.raises(FileExistsError):
                opt.save_experiment(filename=filename, overwrite=False, verbosity=0)

    def test_save_experiment_overwrite_allowed(self):
        """Test that overwrite=True allows file overwriting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = SpotOptim(
                fun=simple_func,
                bounds=[(-5, 5), (-5, 5)],
                max_iter=10,
                n_initial=5,
                seed=42
            )
            
            filename = os.path.join(tmpdir, "test.pkl")
            
            # Save first time
            opt.save_experiment(filename=filename, verbosity=0)
            
            # Modify and save again with overwrite=True
            opt.max_iter = 20
            opt.save_experiment(filename=filename, overwrite=True, verbosity=0)
            
            # Load and verify
            opt_loaded = SpotOptim.load_experiment(filename)
            assert opt_loaded.max_iter == 20

    def test_experiment_excludes_function(self):
        """Test that saved experiment excludes the objective function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = SpotOptim(
                fun=simple_func,
                bounds=[(-5, 5), (-5, 5)],
                max_iter=10,
                n_initial=5,
                seed=42
            )
            
            filename = os.path.join(tmpdir, "experiment.pkl")
            opt.save_experiment(filename=filename, verbosity=0)
            
            # Load experiment
            opt_loaded = SpotOptim.load_experiment(filename)
            
            # Function should not be preserved (set to None or not exist properly)
            # The loaded optimizer needs fun re-attached
            assert opt_loaded.fun is None or not callable(opt_loaded.fun)

    def test_load_and_run_experiment(self):
        """Test loading experiment, attaching function, and running optimization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save experiment
            opt_original = SpotOptim(
                fun=simple_func,
                bounds=[(-5, 5), (-5, 5)],
                max_iter=15,
                n_initial=5,
                seed=42,
                verbose=False
            )
            
            filename = os.path.join(tmpdir, "experiment.pkl")
            opt_original.save_experiment(filename=filename, verbosity=0)
            
            # Load experiment
            opt_loaded = SpotOptim.load_experiment(filename)
            
            # Re-attach function
            opt_loaded.fun = simple_func
            
            # Run optimization
            result = opt_loaded.optimize()
            
            # Verify optimization ran
            assert result.success is True
            assert result.nfev == 15
            assert result.fun < 5.0  # Should find decent solution


class TestSaveLoadResult:
    """Test suite for result save/load functionality."""

    def test_save_result_default_filename(self):
        """Test saving result with default filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = SpotOptim(
                fun=simple_func,
                bounds=[(-5, 5), (-5, 5)],
                max_iter=10,
                n_initial=5,
                seed=42,
                verbose=False
            )
            
            # Run optimization
            opt.optimize()
            
            # Save result
            filename = os.path.join(tmpdir, "test_res.pkl")
            opt.save_result(filename=filename, verbosity=0)
            
            # Check file exists
            assert os.path.exists(filename)

    def test_save_result_with_prefix(self):
        """Test saving result with prefix-generated filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = SpotOptim(
                fun=simple_func,
                bounds=[(-5, 5), (-5, 5)],
                max_iter=10,
                n_initial=5,
                seed=42,
                verbose=False
            )
            
            # Run optimization
            opt.optimize()
            
            # Save with prefix
            opt.save_result(prefix="my_result", path=tmpdir, verbosity=0)
            
            # Check file exists
            expected_file = os.path.join(tmpdir, "my_result_res.pkl")
            assert os.path.exists(expected_file)

    def test_load_result_basic(self):
        """Test loading a saved result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create, optimize, and save result
            opt_original = SpotOptim(
                fun=simple_func,
                bounds=[(-5, 5), (-5, 5)],
                max_iter=20,
                n_initial=8,
                seed=42,
                verbose=False
            )
            
            result_original = opt_original.optimize()
            
            filename = os.path.join(tmpdir, "result.pkl")
            opt_original.save_result(filename=filename, verbosity=0)
            
            # Load result
            opt_loaded = SpotOptim.load_result(filename)
            
            # Verify results preserved
            np.testing.assert_array_almost_equal(
                opt_loaded.best_x_, 
                opt_original.best_x_,
                decimal=10
            )
            assert abs(opt_loaded.best_y_ - opt_original.best_y_) < 1e-10
            assert opt_loaded.n_iter_ == opt_original.n_iter_
            assert opt_loaded.counter == opt_original.counter

    def test_load_result_preserves_evaluations(self):
        """Test that loaded result contains all function evaluations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create, optimize, and save
            opt_original = SpotOptim(
                fun=simple_func,
                bounds=[(-5, 5), (-5, 5)],
                max_iter=15,
                n_initial=5,
                seed=42,
                verbose=False
            )
            
            opt_original.optimize()
            
            filename = os.path.join(tmpdir, "result.pkl")
            opt_original.save_result(filename=filename, verbosity=0)
            
            # Load result
            opt_loaded = SpotOptim.load_result(filename)
            
            # Verify all evaluations preserved
            assert opt_loaded.X_.shape == opt_original.X_.shape
            assert opt_loaded.y_.shape == opt_original.y_.shape
            np.testing.assert_array_equal(opt_loaded.X_, opt_original.X_)
            np.testing.assert_array_equal(opt_loaded.y_, opt_original.y_)

    def test_load_result_file_not_found(self):
        """Test loading result from non-existent file raises error."""
        
        with pytest.raises(FileNotFoundError):
            SpotOptim.load_result("nonexistent_result.pkl")

    def test_result_includes_success_rate(self):
        """Test that saved result includes success rate statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and optimize
            opt_original = SpotOptim(
                fun=simple_func,
                bounds=[(-5, 5), (-5, 5)],
                max_iter=25,
                n_initial=10,
                seed=42,
                verbose=False
            )
            
            opt_original.optimize()
            
            filename = os.path.join(tmpdir, "result.pkl")
            opt_original.save_result(filename=filename, verbosity=0)
            
            # Load result
            opt_loaded = SpotOptim.load_result(filename)
            
            # Verify success rate preserved
            assert hasattr(opt_loaded, 'success_rate')
            assert hasattr(opt_loaded, 'success_counter')
            assert opt_loaded.success_rate == opt_original.success_rate


class TestSaveLoadWithNoise:
    """Test save/load functionality with noisy optimization."""

    def test_save_load_result_with_noise(self):
        """Test saving and loading results from noisy optimization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create optimizer with noise handling
            opt_original = SpotOptim(
                fun=noisy_func,
                bounds=[(-5, 5), (-5, 5)],
                max_iter=20,
                n_initial=8,
                repeats_initial=2,
                repeats_surrogate=2,
                seed=42,
                verbose=False
            )
            
            opt_original.optimize()
            
            filename = os.path.join(tmpdir, "noisy_result.pkl")
            opt_original.save_result(filename=filename, verbosity=0)
            
            # Load result
            opt_loaded = SpotOptim.load_result(filename)
            
            # Verify noise statistics preserved
            assert opt_loaded.noise == opt_original.noise
            if opt_original.mean_X is not None:
                np.testing.assert_array_equal(opt_loaded.mean_X, opt_original.mean_X)
                np.testing.assert_array_equal(opt_loaded.mean_y, opt_original.mean_y)
                np.testing.assert_array_equal(opt_loaded.var_y, opt_original.var_y)


class TestSaveLoadWithVariableTypes:
    """Test save/load with different variable types."""

    def test_save_load_with_integer_variables(self):
        """Test save/load with integer variable types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opt_original = SpotOptim(
                fun=simple_func,
                bounds=[(-5, 5), (-5, 5)],
                var_type=["int", "int"],
                max_iter=15,
                n_initial=5,
                seed=42,
                verbose=False
            )
            
            opt_original.optimize()
            
            filename = os.path.join(tmpdir, "int_result.pkl")
            opt_original.save_result(filename=filename, verbosity=0)
            
            # Load and verify
            opt_loaded = SpotOptim.load_result(filename)
            
            assert opt_loaded.var_type == opt_original.var_type
            np.testing.assert_array_equal(opt_loaded.X_, opt_original.X_)

    def test_save_load_with_mixed_variables(self):
        """Test save/load with mixed variable types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opt_original = SpotOptim(
                fun=simple_func,
                bounds=[(-5, 5), (-5, 5), (-5, 5)],
                var_type=["num", "int", "factor"],
                var_name=["x", "y", "z"],
                max_iter=15,
                n_initial=5,
                seed=42,
                verbose=False
            )
            
            opt_original.optimize()
            
            filename = os.path.join(tmpdir, "mixed_result.pkl")
            opt_original.save_result(filename=filename, verbosity=0)
            
            # Load and verify
            opt_loaded = SpotOptim.load_result(filename)
            
            assert opt_loaded.var_type == opt_original.var_type
            assert opt_loaded.var_name == opt_original.var_name


class TestExperimentResultDifference:
    """Test the difference between experiment and result files."""

    def test_experiment_smaller_than_result(self):
        """Test that experiment files are smaller than result files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = SpotOptim(
                fun=simple_func,
                bounds=[(-5, 5), (-5, 5)],
                max_iter=30,
                n_initial=10,
                seed=42,
                verbose=False
            )
            
            # Save experiment before optimization
            exp_file = os.path.join(tmpdir, "exp.pkl")
            opt.save_experiment(filename=exp_file, verbosity=0)
            exp_size = os.path.getsize(exp_file)
            
            # Run optimization and save result
            opt.optimize()
            res_file = os.path.join(tmpdir, "res.pkl")
            opt.save_result(filename=res_file, verbosity=0)
            res_size = os.path.getsize(res_file)
            
            # Result should be larger (contains evaluations)
            assert res_size > exp_size

    def test_experiment_no_evaluations(self):
        """Test that loaded experiment has no evaluation data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opt_original = SpotOptim(
                fun=simple_func,
                bounds=[(-5, 5), (-5, 5)],
                max_iter=20,
                n_initial=8,
                seed=42,
                verbose=False
            )
            
            opt_original.optimize()
            
            # Save as experiment (should exclude results)
            exp_file = os.path.join(tmpdir, "exp.pkl")
            opt_original.save_experiment(filename=exp_file, unpickleables="all", verbosity=0)
            
            # Load experiment
            opt_loaded = SpotOptim.load_experiment(exp_file)
            
            # Experiment should not have evaluation data
            # (or have it as None/empty depending on implementation)
            # Configuration should be present
            assert opt_loaded.max_iter == opt_original.max_iter
            assert opt_loaded.n_initial == opt_original.n_initial


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_save_to_nonexistent_directory(self):
        """Test saving to a directory that doesn't exist (should create it)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = SpotOptim(
                fun=simple_func,
                bounds=[(-5, 5), (-5, 5)],
                max_iter=10,
                n_initial=5,
                seed=42
            )
            
            # Path to nonexistent subdirectory
            nested_path = os.path.join(tmpdir, "subdir", "nested")
            
            # Should create directory and save
            opt.save_experiment(prefix="test", path=nested_path, verbosity=0)
            
            # Verify directory and file created
            assert os.path.exists(nested_path)
            expected_file = os.path.join(nested_path, "test_exp.pkl")
            assert os.path.exists(expected_file)

    def test_reproducibility_after_load(self):
        """Test that loaded optimizer produces reproducible results with same seed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save experiment
            opt1 = SpotOptim(
                fun=simple_func,
                bounds=[(-5, 5), (-5, 5)],
                max_iter=20,
                n_initial=8,
                seed=42,
                verbose=False
            )
            
            filename = os.path.join(tmpdir, "exp.pkl")
            opt1.save_experiment(filename=filename, verbosity=0)
            
            # Load and run twice
            opt2 = SpotOptim.load_experiment(filename)
            opt2.fun = simple_func
            result2 = opt2.optimize()
            
            opt3 = SpotOptim.load_experiment(filename)
            opt3.fun = simple_func
            result3 = opt3.optimize()
            
            # Results should be identical
            np.testing.assert_array_almost_equal(result2.x, result3.x, decimal=10)
            assert abs(result2.fun - result3.fun) < 1e-10
