# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for LinearRegressor neural network model.
"""

import pytest
import torch
import torch.nn as nn
from spotoptim.nn.linear_regressor import LinearRegressor


class TestLinearRegressorInitialization:
    """Test suite for LinearRegressor initialization."""

    def test_default_initialization(self):
        """Test creating a LinearRegressor with default parameters."""
        model = LinearRegressor(input_dim=10, output_dim=1)

        assert model.input_dim == 10
        assert model.output_dim == 1
        assert model.l1 == 64
        assert model.num_hidden_layers == 0
        assert model.activation_name == "ReLU"

    def test_custom_initialization(self):
        """Test creating a LinearRegressor with custom parameters."""
        model = LinearRegressor(
            input_dim=20, output_dim=5, l1=128, num_hidden_layers=3, activation="Tanh"
        )

        assert model.input_dim == 20
        assert model.output_dim == 5
        assert model.l1 == 128
        assert model.num_hidden_layers == 3
        assert model.activation_name == "Tanh"

    def test_invalid_activation_raises_error(self):
        """Test that invalid activation function raises ValueError."""
        with pytest.raises(
            ValueError, match="Activation function 'InvalidActivation' not found"
        ):
            LinearRegressor(input_dim=10, output_dim=1, activation="InvalidActivation")


class TestLinearRegressorArchitecture:
    """Test suite for network architecture."""

    def test_pure_linear_regression_architecture(self):
        """Test that num_hidden_layers=0 creates pure linear regression."""
        model = LinearRegressor(input_dim=10, output_dim=1, num_hidden_layers=0)

        # Should have only one Linear layer
        linear_layers = [m for m in model.network.modules() if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 1

        # Check dimensions
        assert linear_layers[0].in_features == 10
        assert linear_layers[0].out_features == 1

    def test_single_hidden_layer_architecture(self):
        """Test architecture with one hidden layer."""
        model = LinearRegressor(input_dim=10, output_dim=1, l1=64, num_hidden_layers=1)

        # Should have 2 Linear layers (input->hidden, hidden->output)
        linear_layers = [m for m in model.network.modules() if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 2

        # Check dimensions
        assert linear_layers[0].in_features == 10
        assert linear_layers[0].out_features == 64
        assert linear_layers[1].in_features == 64
        assert linear_layers[1].out_features == 1

        # Should have 1 activation layer
        activation_layers = [
            m for m in model.network.modules() if isinstance(m, nn.ReLU)
        ]
        assert len(activation_layers) == 1

    def test_multiple_hidden_layers_architecture(self):
        """Test architecture with multiple hidden layers."""
        model = LinearRegressor(input_dim=10, output_dim=1, l1=32, num_hidden_layers=3)

        # Should have 4 Linear layers (input->h1, h1->h2, h2->h3, h3->output)
        linear_layers = [m for m in model.network.modules() if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 4

        # Check dimensions
        assert linear_layers[0].in_features == 10
        assert linear_layers[0].out_features == 32
        assert linear_layers[1].in_features == 32
        assert linear_layers[1].out_features == 32
        assert linear_layers[2].in_features == 32
        assert linear_layers[2].out_features == 32
        assert linear_layers[3].in_features == 32
        assert linear_layers[3].out_features == 1

        # Should have 3 activation layers
        activation_layers = [
            m for m in model.network.modules() if isinstance(m, nn.ReLU)
        ]
        assert len(activation_layers) == 3

    def test_multi_output_architecture(self):
        """Test architecture with multiple outputs."""
        model = LinearRegressor(input_dim=5, output_dim=3, l1=16, num_hidden_layers=2)

        linear_layers = [m for m in model.network.modules() if isinstance(m, nn.Linear)]

        # Check final output dimension
        assert linear_layers[-1].out_features == 3


class TestActivationFunctions:
    """Test suite for different activation functions."""

    @pytest.mark.parametrize(
        "activation",
        [
            "ReLU",
            "Sigmoid",
            "Tanh",
            "LeakyReLU",
            "ELU",
            "SELU",
            "GELU",
            "Softplus",
            "Softsign",
        ],
    )
    def test_various_activation_functions(self, activation):
        """Test that various PyTorch activation functions can be used."""
        model = LinearRegressor(
            input_dim=10,
            output_dim=1,
            l1=32,
            num_hidden_layers=2,
            activation=activation,
        )

        assert model.activation_name == activation

        # Check that activation is present in the network
        activation_class = getattr(nn, activation)
        activation_layers = [
            m for m in model.network.modules() if isinstance(m, activation_class)
        ]
        assert (
            len(activation_layers) == 2
        )  # Should have activation for each hidden layer

    def test_relu_activation_by_default(self):
        """Test that ReLU is the default activation."""
        model = LinearRegressor(input_dim=10, output_dim=1, num_hidden_layers=1)

        relu_layers = [m for m in model.network.modules() if isinstance(m, nn.ReLU)]
        assert len(relu_layers) == 1


class TestForwardPass:
    """Test suite for forward pass functionality."""

    def test_forward_pass_output_shape(self):
        """Test that forward pass produces correct output shape."""
        model = LinearRegressor(input_dim=10, output_dim=1, num_hidden_layers=2)

        # Create random input
        x = torch.randn(32, 10)  # Batch of 32 samples

        # Forward pass
        output = model(x)

        # Check output shape
        assert output.shape == (32, 1)

    def test_forward_pass_multi_output(self):
        """Test forward pass with multiple outputs."""
        model = LinearRegressor(input_dim=5, output_dim=3, l1=16, num_hidden_layers=1)

        x = torch.randn(16, 5)
        output = model(x)

        assert output.shape == (16, 3)

    def test_forward_pass_single_sample(self):
        """Test forward pass with single sample."""
        model = LinearRegressor(input_dim=10, output_dim=1, num_hidden_layers=1)

        x = torch.randn(1, 10)
        output = model(x)

        assert output.shape == (1, 1)

    def test_forward_pass_pure_linear(self):
        """Test forward pass with no hidden layers."""
        model = LinearRegressor(input_dim=5, output_dim=2, num_hidden_layers=0)

        x = torch.randn(10, 5)
        output = model(x)

        assert output.shape == (10, 2)

    def test_forward_pass_numerical_output(self):
        """Test that forward pass produces finite numerical outputs."""
        model = LinearRegressor(input_dim=10, output_dim=1, l1=32, num_hidden_layers=2)

        x = torch.randn(5, 10)
        output = model(x)

        # Check that output is finite (no NaN or Inf)
        assert torch.isfinite(output).all()


class TestGradientFlow:
    """Test suite for gradient flow in the model."""

    def test_gradients_flow_through_network(self):
        """Test that gradients flow through the network during backpropagation."""
        model = LinearRegressor(input_dim=10, output_dim=1, l1=32, num_hidden_layers=2)

        # Create input and target
        x = torch.randn(5, 10, requires_grad=True)
        target = torch.randn(5, 1)

        # Forward pass
        output = model(x)

        # Compute loss
        loss = nn.MSELoss()(output, target)

        # Backward pass
        loss.backward()

        # Check that gradients exist for all parameters
        for param in model.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()

    def test_zero_grad_clears_gradients(self):
        """Test that zero_grad properly clears gradients."""
        model = LinearRegressor(input_dim=10, output_dim=1, num_hidden_layers=1)

        x = torch.randn(5, 10)
        target = torch.randn(5, 1)

        # First pass
        output = model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()

        # Clear gradients
        model.zero_grad()

        # Check all gradients are None or zero
        for param in model.parameters():
            if param.grad is not None:
                assert (param.grad == 0).all()


class TestModelTraining:
    """Test suite for model training capability."""

    def test_simple_training_loop(self):
        """Test that model can be trained with a simple training loop."""
        # Create a simple linear dataset: y = 2x + 1
        torch.manual_seed(42)
        X_train = torch.randn(100, 1)
        y_train = 2 * X_train + 1 + 0.1 * torch.randn(100, 1)

        # Create model
        model = LinearRegressor(input_dim=1, output_dim=1, num_hidden_layers=0)

        # Training setup
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Initial loss
        initial_loss = criterion(model(X_train), y_train).item()

        # Train for a few epochs
        for epoch in range(50):
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

        # Final loss
        final_loss = criterion(model(X_train), y_train).item()

        # Loss should decrease
        assert final_loss < initial_loss

    def test_training_with_hidden_layers(self):
        """Test training with hidden layers on a nonlinear function."""
        torch.manual_seed(42)

        # Nonlinear function: y = x^2
        X_train = torch.linspace(-3, 3, 100).unsqueeze(1)
        y_train = X_train**2

        # Create model with hidden layers
        model = LinearRegressor(
            input_dim=1, output_dim=1, l1=16, num_hidden_layers=2, activation="Tanh"
        )

        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Initial loss
        initial_loss = criterion(model(X_train), y_train).item()

        # Train
        for epoch in range(100):
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

        # Final loss
        final_loss = criterion(model(X_train), y_train).item()

        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.5


class TestModelPersistence:
    """Test suite for model saving and loading."""

    def test_state_dict_save_load(self):
        """Test that model can be saved and loaded using state_dict."""
        # Create and initialize model
        model1 = LinearRegressor(input_dim=10, output_dim=1, l1=32, num_hidden_layers=2)

        # Save state
        state_dict = model1.state_dict()

        # Create new model and load state
        model2 = LinearRegressor(input_dim=10, output_dim=1, l1=32, num_hidden_layers=2)
        model2.load_state_dict(state_dict)

        # Test that both models produce same output
        x = torch.randn(5, 10)
        output1 = model1(x)
        output2 = model2(x)

        assert torch.allclose(output1, output2)


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_single_neuron_hidden_layer(self):
        """Test with single neuron in hidden layer."""
        model = LinearRegressor(input_dim=10, output_dim=1, l1=1, num_hidden_layers=1)

        x = torch.randn(5, 10)
        output = model(x)

        assert output.shape == (5, 1)
        assert torch.isfinite(output).all()

    def test_very_deep_network(self):
        """Test with many hidden layers."""
        model = LinearRegressor(input_dim=10, output_dim=1, l1=16, num_hidden_layers=10)

        # Count layers
        linear_layers = [m for m in model.network.modules() if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 11  # 10 hidden + 1 output

        # Test forward pass
        x = torch.randn(5, 10)
        output = model(x)
        assert output.shape == (5, 1)

    def test_large_batch_size(self):
        """Test with large batch size."""
        model = LinearRegressor(input_dim=10, output_dim=1, l1=32, num_hidden_layers=2)

        x = torch.randn(1000, 10)
        output = model(x)

        assert output.shape == (1000, 1)
        assert torch.isfinite(output).all()

    def test_high_dimensional_input(self):
        """Test with high dimensional input."""
        model = LinearRegressor(input_dim=500, output_dim=1, l1=64, num_hidden_layers=1)

        x = torch.randn(10, 500)
        output = model(x)

        assert output.shape == (10, 1)

    def test_high_dimensional_output(self):
        """Test with high dimensional output."""
        model = LinearRegressor(
            input_dim=10, output_dim=100, l1=32, num_hidden_layers=1
        )

        x = torch.randn(5, 10)
        output = model(x)

        assert output.shape == (5, 100)


class TestOptimizer:
    """Test suite for optimizer functionality."""

    def test_get_optimizer_default(self):
        """Test getting default Adam optimizer."""
        model = LinearRegressor(input_dim=10, output_dim=1)
        optimizer = model.get_optimizer()

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.defaults["lr"] == 0.001

    def test_get_optimizer_custom_learning_rate(self):
        """Test getting optimizer with custom learning rate (unified lr mapping)."""
        model = LinearRegressor(input_dim=10, output_dim=1)
        # lr=0.01 is unified lr, mapped to 0.01 * 0.001 = 0.00001 for Adam
        optimizer = model.get_optimizer("Adam", lr=0.01)

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.defaults["lr"] == 0.01 * 0.001  # 0.00001

    @pytest.mark.parametrize(
        "optimizer_name,optimizer_class",
        [
            ("Adam", torch.optim.Adam),
            ("AdamW", torch.optim.AdamW),
            ("Adamax", torch.optim.Adamax),
            ("SGD", torch.optim.SGD),
            ("RMSprop", torch.optim.RMSprop),
            ("Adagrad", torch.optim.Adagrad),
            ("Adadelta", torch.optim.Adadelta),
            ("NAdam", torch.optim.NAdam),
            ("RAdam", torch.optim.RAdam),
        ],
    )
    def test_various_optimizers(self, optimizer_name, optimizer_class):
        """Test that various PyTorch optimizers can be instantiated."""
        model = LinearRegressor(input_dim=10, output_dim=1, num_hidden_layers=1)
        optimizer = model.get_optimizer(optimizer_name)

        assert isinstance(optimizer, optimizer_class)

    def test_sgd_with_momentum(self):
        """Test SGD optimizer with momentum parameter (unified lr mapping)."""
        model = LinearRegressor(input_dim=10, output_dim=1)
        # lr=0.01 is unified lr, mapped to 0.01 * 0.01 = 0.0001 for SGD
        optimizer = model.get_optimizer("SGD", lr=0.01, momentum=0.9)

        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.defaults["lr"] == 0.01 * 0.01  # 0.0001
        assert optimizer.defaults["momentum"] == 0.9

    def test_adamw_with_weight_decay(self):
        """Test AdamW optimizer with weight decay."""
        model = LinearRegressor(input_dim=10, output_dim=1)
        optimizer = model.get_optimizer("AdamW", lr=0.001, weight_decay=0.01)

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["weight_decay"] == 0.01

    def test_rmsprop_with_alpha(self):
        """Test RMSprop with custom alpha parameter."""
        model = LinearRegressor(input_dim=10, output_dim=1)
        optimizer = model.get_optimizer("RMSprop", lr=0.01, alpha=0.95)

        assert isinstance(optimizer, torch.optim.RMSprop)
        assert optimizer.defaults["alpha"] == 0.95

    def test_invalid_optimizer_raises_error(self):
        """Test that invalid optimizer name raises ValueError."""
        model = LinearRegressor(input_dim=10, output_dim=1)

        with pytest.raises(ValueError, match="Optimizer 'InvalidOptimizer' not found"):
            model.get_optimizer("InvalidOptimizer")

    def test_optimizer_has_model_parameters(self):
        """Test that optimizer is properly initialized with model parameters."""
        model = LinearRegressor(input_dim=10, output_dim=1, num_hidden_layers=2)
        optimizer = model.get_optimizer("Adam")

        # Check that optimizer has parameter groups
        assert len(optimizer.param_groups) > 0

        # Check that parameters match model parameters
        model_params = list(model.parameters())
        optimizer_params = []
        for param_group in optimizer.param_groups:
            optimizer_params.extend(param_group["params"])

        assert len(model_params) == len(optimizer_params)

    def test_training_with_get_optimizer(self):
        """Test that optimizer from get_optimizer can be used for training."""
        torch.manual_seed(42)

        # Simple dataset
        X = torch.randn(100, 5)
        y = torch.randn(100, 1)

        # Create model and optimizer
        model = LinearRegressor(input_dim=5, output_dim=1, num_hidden_layers=1)
        optimizer = model.get_optimizer("Adam", lr=0.01)
        criterion = nn.MSELoss()

        # Initial loss
        initial_loss = criterion(model(X), y).item()

        # Train for a few steps
        for _ in range(20):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # Final loss
        final_loss = criterion(model(X), y).item()

        # Loss should decrease
        assert final_loss < initial_loss

    def test_different_optimizers_training(self):
        """Test training with different optimizers."""
        torch.manual_seed(42)

        X = torch.randn(50, 3)
        y = torch.randn(50, 1)

        optimizers_to_test = ["Adam", "SGD", "RMSprop", "AdamW"]

        for opt_name in optimizers_to_test:
            model = LinearRegressor(input_dim=3, output_dim=1, num_hidden_layers=1)

            if opt_name == "SGD":
                optimizer = model.get_optimizer(opt_name, lr=0.01, momentum=0.9)
            else:
                optimizer = model.get_optimizer(opt_name, lr=0.01)

            criterion = nn.MSELoss()

            initial_loss = criterion(model(X), y).item()

            # Train
            for _ in range(10):
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()

            final_loss = criterion(model(X), y).item()

            # Loss should decrease for all optimizers
            assert final_loss < initial_loss, f"{opt_name} failed to reduce loss"
