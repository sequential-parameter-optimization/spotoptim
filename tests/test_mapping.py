"""Tests for learning rate mapping utilities."""

import pytest
from spotoptim.utils.mapping import map_lr, OPTIMIZER_DEFAULT_LR


class TestMapLR:
    """Test suite for map_lr function."""

    def test_default_lr_values(self):
        """Test that lr_unified=1.0 returns optimizer defaults."""
        for optimizer_name, expected_lr in OPTIMIZER_DEFAULT_LR.items():
            actual_lr = map_lr(1.0, optimizer_name)
            assert (
                actual_lr == expected_lr
            ), f"For {optimizer_name}, expected {expected_lr}, got {actual_lr}"

    def test_scaling_multiplier(self):
        """Test that lr_unified acts as a multiplier."""
        # Test with Adam (default 0.001)
        assert map_lr(1.0, "Adam") == 0.001
        assert map_lr(2.0, "Adam") == 0.002
        assert map_lr(0.5, "Adam") == 0.0005
        assert map_lr(10.0, "Adam") == 0.01
        assert map_lr(0.1, "Adam") == 0.0001

        # Test with SGD (default 0.01)
        assert map_lr(1.0, "SGD") == 0.01
        assert map_lr(2.0, "SGD") == 0.02
        assert map_lr(0.5, "SGD") == 0.005
        assert map_lr(10.0, "SGD") == 0.1
        assert map_lr(0.1, "SGD") == 0.001

        # Test with Adadelta (default 1.0)
        assert map_lr(1.0, "Adadelta") == 1.0
        assert map_lr(2.0, "Adadelta") == 2.0
        assert map_lr(0.5, "Adadelta") == 0.5

    def test_all_optimizers(self):
        """Test that all supported optimizers work."""
        lr_unified = 0.5
        for optimizer_name in OPTIMIZER_DEFAULT_LR.keys():
            try:
                lr = map_lr(lr_unified, optimizer_name)
                assert lr > 0, f"{optimizer_name} returned non-positive lr: {lr}"
                # Check that it's the correct multiple
                expected = lr_unified * OPTIMIZER_DEFAULT_LR[optimizer_name]
                assert (
                    lr == expected
                ), f"{optimizer_name}: expected {expected}, got {lr}"
            except Exception as e:
                pytest.fail(f"Failed for {optimizer_name}: {str(e)}")

    def test_no_scaling(self):
        """Test use_default_scale=False returns lr_unified directly."""
        lr_unified = 0.123
        for optimizer_name in OPTIMIZER_DEFAULT_LR.keys():
            actual_lr = map_lr(lr_unified, optimizer_name, use_default_scale=False)
            assert (
                actual_lr == lr_unified
            ), f"With use_default_scale=False, expected {lr_unified}, got {actual_lr}"

    def test_invalid_optimizer(self):
        """Test that invalid optimizer name raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            map_lr(0.01, "InvalidOptimizer")

        with pytest.raises(ValueError, match="not supported"):
            map_lr(0.01, "adam")  # Case sensitive

        with pytest.raises(ValueError, match="not supported"):
            map_lr(0.01, "")

    def test_invalid_learning_rate(self):
        """Test that non-positive learning rates raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            map_lr(0.0, "Adam")

        with pytest.raises(ValueError, match="must be positive"):
            map_lr(-0.01, "Adam")

        with pytest.raises(ValueError, match="must be positive"):
            map_lr(-1.0, "SGD")

    def test_extreme_values(self):
        """Test with extreme but valid learning rate values."""
        # Very small unified lr
        lr = map_lr(1e-6, "Adam")
        assert lr == 1e-6 * 0.001
        assert lr > 0

        # Very large unified lr
        lr = map_lr(1000.0, "Adam")
        assert lr == 1000.0 * 0.001
        assert lr == 1.0

        # Test with SGD
        lr = map_lr(1e-6, "SGD")
        assert lr == 1e-6 * 0.01
        assert lr > 0

    def test_log_scale_conversion(self):
        """Test typical log-scale hyperparameter optimization scenario."""
        # Common pattern: sample from log10 scale [-4, 0]
        # Then convert: lr_unified = 10^x

        # x = -4 → lr_unified = 0.0001
        lr_unified = 10 ** (-4)
        lr_adam = map_lr(lr_unified, "Adam")
        assert lr_adam == 0.0001 * 0.001  # 1e-7

        # x = -2 → lr_unified = 0.01
        lr_unified = 10 ** (-2)
        lr_sgd = map_lr(lr_unified, "SGD")
        assert lr_sgd == 0.01 * 0.01  # 1e-4

        # x = 0 → lr_unified = 1.0
        lr_unified = 10**0
        lr_rmsprop = map_lr(lr_unified, "RMSprop")
        assert lr_rmsprop == 1.0 * 0.01  # 0.01 (default)

    def test_relative_scaling_consistency(self):
        """Test that relative scaling is consistent across optimizers."""
        lr_base = 1.0
        lr_double = 2.0
        lr_half = 0.5

        for optimizer_name in OPTIMIZER_DEFAULT_LR.keys():
            base = map_lr(lr_base, optimizer_name)
            double = map_lr(lr_double, optimizer_name)
            half = map_lr(lr_half, optimizer_name)

            # Check that doubling unified lr doubles actual lr
            assert abs(double - 2 * base) < 1e-10, f"{optimizer_name}: doubling failed"

            # Check that halving unified lr halves actual lr
            assert abs(half - 0.5 * base) < 1e-10, f"{optimizer_name}: halving failed"

    def test_adam_family_consistency(self):
        """Test that Adam-family optimizers have consistent defaults."""
        adam_family = ["Adam", "AdamW", "RAdam", "SparseAdam"]
        lr_unified = 1.0

        adam_lr = map_lr(lr_unified, "Adam")

        for optimizer_name in adam_family:
            if optimizer_name == "Adam":
                continue
            lr = map_lr(lr_unified, optimizer_name)
            # Adam variants should have similar (though not necessarily identical) defaults
            # We just check they're in the same ballpark (within 10x)
            ratio = lr / adam_lr
            assert (
                0.1 <= ratio <= 10
            ), f"{optimizer_name} lr {lr} too different from Adam lr {adam_lr}"

    def test_float_precision(self):
        """Test that float precision is maintained."""
        # Test with values that could cause floating point issues
        lr_unified = 0.123456789
        lr = map_lr(lr_unified, "Adam")
        expected = lr_unified * 0.001
        assert abs(lr - expected) < 1e-15

    def test_optimizer_name_case_sensitive(self):
        """Test that optimizer names are case-sensitive."""
        # Correct case should work
        lr = map_lr(1.0, "Adam")
        assert lr == 0.001

        # Wrong case should fail
        with pytest.raises(ValueError):
            map_lr(1.0, "adam")

        with pytest.raises(ValueError):
            map_lr(1.0, "ADAM")

    def test_all_pytorch_optimizers_covered(self):
        """Test that all major PyTorch optimizers are covered."""
        expected_optimizers = [
            "Adadelta",
            "Adagrad",
            "Adam",
            "AdamW",
            "SparseAdam",
            "Adamax",
            "ASGD",
            "LBFGS",
            "NAdam",
            "RAdam",
            "RMSprop",
            "Rprop",
            "SGD",
        ]

        for optimizer_name in expected_optimizers:
            assert (
                optimizer_name in OPTIMIZER_DEFAULT_LR
            ), f"Optimizer {optimizer_name} missing from OPTIMIZER_DEFAULT_LR"

    def test_realistic_use_case(self):
        """Test realistic hyperparameter optimization scenario."""
        # Scenario: Optimize over different optimizers with unified lr scale
        optimizers = ["Adam", "SGD", "RMSprop", "AdamW"]
        lr_unified_candidates = [0.1, 0.5, 1.0, 2.0, 5.0]

        for opt in optimizers:
            for lr_u in lr_unified_candidates:
                lr = map_lr(lr_u, opt)

                # Check that lr is reasonable
                assert (
                    1e-6 < lr < 10.0
                ), f"Unreasonable lr {lr} for {opt} with unified lr {lr_u}"

                # Check that it's the correct mapping
                expected = lr_u * OPTIMIZER_DEFAULT_LR[opt]
                assert abs(lr - expected) < 1e-10


class TestDefaultLRConstants:
    """Test suite for OPTIMIZER_DEFAULT_LR constants."""

    def test_all_values_positive(self):
        """Test that all default learning rates are positive."""
        for optimizer_name, lr in OPTIMIZER_DEFAULT_LR.items():
            assert lr > 0, f"{optimizer_name} has non-positive default lr: {lr}"

    def test_all_values_reasonable(self):
        """Test that all default learning rates are in reasonable range."""
        for optimizer_name, lr in OPTIMIZER_DEFAULT_LR.items():
            assert (
                0.0001 <= lr <= 10.0
            ), f"{optimizer_name} has unreasonable default lr: {lr}"

    def test_dictionary_not_empty(self):
        """Test that the dictionary is not empty."""
        assert len(OPTIMIZER_DEFAULT_LR) > 0

    def test_values_match_pytorch_defaults(self):
        """Test that values match PyTorch documentation."""
        # These values are from PyTorch 2.x documentation
        # https://pytorch.org/docs/stable/optim.html
        expected_defaults = {
            "Adadelta": 1.0,
            "Adagrad": 0.01,
            "Adam": 0.001,
            "AdamW": 0.001,
            "SparseAdam": 0.001,
            "Adamax": 0.002,
            "ASGD": 0.01,
            "LBFGS": 1.0,
            "NAdam": 0.002,
            "RAdam": 0.001,
            "RMSprop": 0.01,
            "Rprop": 0.01,
            "SGD": 0.01,
        }

        for optimizer_name, expected_lr in expected_defaults.items():
            actual_lr = OPTIMIZER_DEFAULT_LR[optimizer_name]
            assert actual_lr == expected_lr, (
                f"{optimizer_name}: expected default {expected_lr}, "
                f"got {actual_lr}. Please verify with PyTorch documentation."
            )
