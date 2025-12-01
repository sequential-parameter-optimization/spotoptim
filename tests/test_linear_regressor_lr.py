"""Tests for LinearRegressor with learning rate mapping integration."""

import pytest
import torch
import torch.nn as nn
from spotoptim.nn.linear_regressor import LinearRegressor
from spotoptim.utils.mapping import OPTIMIZER_DEFAULT_LR


class TestLinearRegressorLR:
    """Test suite for LinearRegressor lr parameter and integration with map_lr."""

    def test_default_lr_attribute(self):
        """Test that model has lr attribute with default value."""
        model = LinearRegressor(input_dim=10, output_dim=1)
        assert hasattr(model, "lr")
        assert model.lr == 1.0

    def test_custom_lr_attribute(self):
        """Test that custom lr is stored correctly."""
        model = LinearRegressor(input_dim=10, output_dim=1, lr=0.5)
        assert model.lr == 0.5

        model = LinearRegressor(input_dim=10, output_dim=1, lr=10.0)
        assert model.lr == 10.0

    def test_optimizer_with_model_lr(self):
        """Test that get_optimizer uses model's lr when not specified."""
        model = LinearRegressor(input_dim=10, output_dim=1, lr=2.0)

        # Adam: default 0.001, so 2.0 * 0.001 = 0.002
        optimizer = model.get_optimizer("Adam")
        assert optimizer.param_groups[0]["lr"] == 0.002

        # SGD: default 0.01, so 2.0 * 0.01 = 0.02
        optimizer = model.get_optimizer("SGD")
        assert optimizer.param_groups[0]["lr"] == 0.02

    def test_optimizer_with_override_lr(self):
        """Test that lr parameter overrides model's lr."""
        model = LinearRegressor(input_dim=10, output_dim=1, lr=1.0)

        # Override with lr=0.5
        optimizer = model.get_optimizer("Adam", lr=0.5)
        assert optimizer.param_groups[0]["lr"] == 0.5 * 0.001  # 0.0005

        optimizer = model.get_optimizer("SGD", lr=0.5)
        assert optimizer.param_groups[0]["lr"] == 0.5 * 0.01  # 0.005

    def test_unified_lr_across_optimizers(self):
        """Test that same unified lr gives different actual lrs for different optimizers."""
        model = LinearRegressor(input_dim=10, output_dim=1, lr=1.0)

        adam_opt = model.get_optimizer("Adam")
        sgd_opt = model.get_optimizer("SGD")
        rmsprop_opt = model.get_optimizer("RMSprop")

        # With unified lr=1.0, each should get its default
        assert adam_opt.param_groups[0]["lr"] == 0.001
        assert sgd_opt.param_groups[0]["lr"] == 0.01
        assert rmsprop_opt.param_groups[0]["lr"] == 0.01

    def test_all_supported_optimizers(self):
        """Test that all map_lr supported optimizers work."""
        model = LinearRegressor(input_dim=10, output_dim=1, lr=1.0)

        for optimizer_name in OPTIMIZER_DEFAULT_LR.keys():
            try:
                optimizer = model.get_optimizer(optimizer_name)
                expected_lr = OPTIMIZER_DEFAULT_LR[optimizer_name]
                actual_lr = optimizer.param_groups[0]["lr"]
                assert (
                    actual_lr == expected_lr
                ), f"{optimizer_name}: expected {expected_lr}, got {actual_lr}"
            except Exception as e:
                pytest.fail(f"Failed to create {optimizer_name}: {str(e)}")

    def test_optimizer_with_additional_kwargs(self):
        """Test that additional optimizer kwargs are passed correctly."""
        model = LinearRegressor(input_dim=10, output_dim=1, lr=1.0)

        # SGD with momentum
        optimizer = model.get_optimizer("SGD", momentum=0.9)
        assert optimizer.param_groups[0]["momentum"] == 0.9
        assert optimizer.param_groups[0]["lr"] == 0.01  # default

        # AdamW with weight_decay
        optimizer = model.get_optimizer("AdamW", weight_decay=0.01)
        assert optimizer.param_groups[0]["weight_decay"] == 0.01
        assert optimizer.param_groups[0]["lr"] == 0.001  # default

        # RMSprop with alpha
        optimizer = model.get_optimizer("RMSprop", alpha=0.95)
        assert optimizer.param_groups[0]["alpha"] == 0.95
        assert optimizer.param_groups[0]["lr"] == 0.01  # default

    def test_training_with_mapped_lr(self):
        """Test that training works with mapped learning rates."""
        torch.manual_seed(42)

        # Create simple dataset
        X = torch.randn(100, 5)
        y = torch.randn(100, 1)

        # Create model with unified lr
        model = LinearRegressor(
            input_dim=5, output_dim=1, l1=8, num_hidden_layers=1, lr=1.0
        )
        optimizer = model.get_optimizer("Adam")
        criterion = nn.MSELoss()

        # Initial loss
        model.eval()
        with torch.no_grad():
            initial_loss = criterion(model(X), y).item()

        # Train for a few steps
        model.train()
        for _ in range(50):
            optimizer.zero_grad()
            predictions = model(X)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

        # Final loss should be lower
        model.eval()
        with torch.no_grad():
            final_loss = criterion(model(X), y).item()

        assert final_loss < initial_loss, "Training should reduce loss"

    def test_lr_scaling_consistency(self):
        """Test that lr scaling is consistent."""
        model = LinearRegressor(input_dim=10, output_dim=1)

        # Test with different unified lr values
        for lr_unified in [0.1, 0.5, 1.0, 2.0, 10.0]:
            model.lr = lr_unified

            optimizer = model.get_optimizer("Adam")
            expected = lr_unified * 0.001
            actual = optimizer.param_groups[0]["lr"]
            assert abs(actual - expected) < 1e-10

    def test_zero_hidden_layers_with_lr(self):
        """Test that lr works with pure linear regression."""
        model = LinearRegressor(input_dim=10, output_dim=1, num_hidden_layers=0, lr=0.5)
        optimizer = model.get_optimizer("Adam")

        # Should use 0.5 * 0.001 = 0.0005
        assert optimizer.param_groups[0]["lr"] == 0.0005

    def test_different_activations_with_lr(self):
        """Test that lr works with different activation functions."""
        activations = ["ReLU", "Tanh", "Sigmoid", "LeakyReLU"]

        for activation in activations:
            model = LinearRegressor(
                input_dim=10,
                output_dim=1,
                l1=16,
                num_hidden_layers=1,
                activation=activation,
                lr=1.0,
            )
            optimizer = model.get_optimizer("Adam")
            assert optimizer.param_groups[0]["lr"] == 0.001

    def test_log_scale_lr_optimization(self):
        """Test typical log-scale hyperparameter optimization scenario."""
        # Common pattern: sample from log10 scale [-4, 0], then lr_unified = 10^x

        for log_lr in [-4, -3, -2, -1, 0]:
            lr_unified = 10**log_lr
            model = LinearRegressor(input_dim=10, output_dim=1, lr=lr_unified)

            optimizer = model.get_optimizer("Adam")
            expected = lr_unified * 0.001
            actual = optimizer.param_groups[0]["lr"]
            assert abs(actual - expected) < 1e-15

    def test_lr_with_multiple_optimizer_changes(self):
        """Test that same model can create optimizers with different names."""
        model = LinearRegressor(input_dim=10, output_dim=1, lr=1.0)

        # Create multiple optimizers with same model
        adam_opt = model.get_optimizer("Adam")
        sgd_opt = model.get_optimizer("SGD")
        rmsprop_opt = model.get_optimizer("RMSprop")

        # Each should have correct lr
        assert adam_opt.param_groups[0]["lr"] == 0.001
        assert sgd_opt.param_groups[0]["lr"] == 0.01
        assert rmsprop_opt.param_groups[0]["lr"] == 0.01

    def test_backward_compatibility(self):
        """Test that old code without lr parameter still works."""
        # Old code that doesn't specify lr
        model = LinearRegressor(
            input_dim=10, output_dim=1, l1=32, num_hidden_layers=2, activation="ReLU"
        )

        # Should default to lr=1.0
        assert model.lr == 1.0

        # Should still work with old-style optimizer creation
        optimizer = model.get_optimizer("Adam")
        assert optimizer.param_groups[0]["lr"] == 0.001


class TestLinearRegressorLRIntegration:
    """Integration tests for LinearRegressor with learning rate mapping."""

    def test_full_training_cycle_adam(self):
        """Test complete training cycle with Adam and mapped lr."""
        torch.manual_seed(42)

        X = torch.randn(200, 10)
        y = torch.randn(200, 1)

        model = LinearRegressor(
            input_dim=10, output_dim=1, l1=16, num_hidden_layers=1, lr=5.0
        )
        optimizer = model.get_optimizer("Adam")  # Uses 5.0 * 0.001 = 0.005
        criterion = nn.MSELoss()

        # Training
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            predictions = model(X)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

        # Check that optimizer's lr is correct
        assert optimizer.param_groups[0]["lr"] == 0.005

    def test_full_training_cycle_sgd(self):
        """Test complete training cycle with SGD and mapped lr."""
        torch.manual_seed(42)

        X = torch.randn(200, 10)
        y = torch.randn(200, 1)

        model = LinearRegressor(
            input_dim=10, output_dim=1, l1=16, num_hidden_layers=1, lr=0.5
        )
        optimizer = model.get_optimizer("SGD", momentum=0.9)  # Uses 0.5 * 0.01 = 0.005
        criterion = nn.MSELoss()

        # Training
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            predictions = model(X)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

        # Check that optimizer's lr is correct
        assert optimizer.param_groups[0]["lr"] == 0.005

    def test_optimizer_comparison_with_unified_lr(self):
        """Test that unified lr enables fair optimizer comparison."""
        torch.manual_seed(42)

        X = torch.randn(100, 10)
        y = torch.randn(100, 1)

        lr_unified = 1.0
        results = {}

        for optimizer_name in ["Adam", "SGD", "RMSprop"]:
            torch.manual_seed(42)  # Reset for fair comparison

            model = LinearRegressor(
                input_dim=10, output_dim=1, l1=16, num_hidden_layers=1, lr=lr_unified
            )
            optimizer = model.get_optimizer(optimizer_name)
            criterion = nn.MSELoss()

            # Train
            model.train()
            for _ in range(50):
                optimizer.zero_grad()
                predictions = model(X)
                loss = criterion(predictions, y)
                loss.backward()
                optimizer.step()

            # Evaluate
            model.eval()
            with torch.no_grad():
                final_loss = criterion(model(X), y).item()

            results[optimizer_name] = final_loss

        # All optimizers should have produced reasonable results
        for optimizer_name, loss in results.items():
            assert loss > 0, f"{optimizer_name} produced non-positive loss"
            assert loss < float("inf"), f"{optimizer_name} diverged"

    def test_realistic_hyperparameter_optimization_scenario(self):
        """Test realistic scenario with multiple hyperparameters."""
        torch.manual_seed(42)

        X_train = torch.randn(100, 10)
        y_train = torch.randn(100, 1)
        X_test = torch.randn(30, 10)
        y_test = torch.randn(30, 1)

        # Simulate hyperparameter optimization over lr and optimizer
        configs = [
            (0.1, "Adam"),
            (0.5, "Adam"),
            (1.0, "Adam"),
            (0.1, "SGD"),
            (0.5, "SGD"),
            (1.0, "SGD"),
        ]

        for lr_unified, optimizer_name in configs:
            model = LinearRegressor(
                input_dim=10, output_dim=1, l1=16, num_hidden_layers=1, lr=lr_unified
            )
            # Only pass momentum for SGD
            if optimizer_name == "SGD":
                optimizer = model.get_optimizer(optimizer_name, momentum=0.9)
            else:
                optimizer = model.get_optimizer(optimizer_name)
            criterion = nn.MSELoss()

            # Train
            model.train()
            for _ in range(30):
                optimizer.zero_grad()
                predictions = model(X_train)
                loss = criterion(predictions, y_train)
                loss.backward()
                optimizer.step()

            # Evaluate
            model.eval()
            with torch.no_grad():
                test_loss = criterion(model(X_test), y_test).item()

            # Should produce valid results
            assert test_loss > 0
            assert test_loss < float("inf")
