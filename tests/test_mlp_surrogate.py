
import numpy as np
import pytest
import torch
from spotoptim.surrogate import MLPSurrogate

def test_mlp_surrogate_initialization():
    """Test default initialization."""
    mlp = MLPSurrogate()
    assert mlp.name == "MLPSurrogate"
    assert mlp.optimizer_name == "AdamWScheduleFree"
    assert mlp.l1 == 128
    assert mlp.num_hidden_layers == 3
    assert mlp.dropout == 0.0
    assert mlp.mc_dropout_passes == 30

def test_mlp_surrogate_fit_predict_shape():
    """Test that fit and predict handle shapes correctly."""
    X = np.random.rand(10, 2)
    y = np.random.rand(10)
    
    mlp = MLPSurrogate(epochs=1) # Minimal training
    mlp.fit(X, y)
    
    # Predict
    pred = mlp.predict(X)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (10,)
    assert pred.dtype == np.float32 or pred.dtype == np.float64

def test_mlp_surrogate_learning_simple_function():
    """Test that it can learn a simple quadratic function."""
    # X in [-1, 1], y = X^2
    X = np.linspace(-1, 1, 50).reshape(-1, 1)
    y = X**2
    
    # Needs sufficient capacity and training
    mlp = MLPSurrogate(
        l1=64, 
        num_hidden_layers=2, 
        dropout=0.0, # Determine deterministic fit first 
        lr=0.01, 
        epochs=1000, 
        seed=42
    )
    mlp.fit(X, y)
    
    pred = mlp.predict(X)
    mse = np.mean((pred - y.flatten())**2)
    
    # Should fit reasonably well
    assert mse < 0.1, f"MSE {mse} is too high for simple quadratic"

def test_mlp_surrogate_uncertainty_estimation():
    """Test that return_std=True returns uncertainty when dropout is used."""
    X = np.linspace(0, 1, 20).reshape(-1, 1)
    y = X
    
    # Set high dropout and few epochs so it doesn't converge perfectly to 0 error
    # and maintains variance in dropout masks
    mlp = MLPSurrogate(dropout=0.5, epochs=10, mc_dropout_passes=20)
    mlp.fit(X, y)
    
    mean, std = mlp.predict(X, return_std=True)
    
    assert mean.shape == (20,)
    assert std.shape == (20,)
    # With dropout=0.5, we expect non-zero std deviation
    assert np.all(std >= 0)
    assert np.mean(std) > 0, "Expected non-zero uncertainty with dropout"

def test_compatibility_with_var_type():
    """Test it accepts var_type argument without crashing."""
    X = np.array([[0.0, 1], [0.5, 2], [1.0, 3]])
    y = np.array([0.0, 0.5, 1.0])
    var_type = ["float", "int"]
    
    mlp = MLPSurrogate(var_type=var_type, epochs=1)
    mlp.fit(X, y)
    pred = mlp.predict(X)
    assert pred.shape == (3,)

def test_multidimensional_output_support():
    """Test support for multi-output (though SpotOptim main path uses single objective)."""
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 2) # Two outputs
    
    mlp = MLPSurrogate(epochs=1)
    mlp.fit(X, y)
    
    pred = mlp.predict(X)
    assert pred.shape == (10, 2)
    
    # With std
    mean, std = mlp.predict(X, return_std=True)
    assert mean.shape == (10, 2)
    assert std.shape == (10, 2)
