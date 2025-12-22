import numpy as np
import pytest
from spotoptim.sampling.poor import Poor

def test_poor_initialization():
    """Test initialization of Poor class."""
    poor = Poor(k=2, seed=42)
    assert poor.k == 2
    assert poor.seed == 42
    
    # Test default values
    poor_default = Poor()
    assert poor_default.k == 2
    assert poor_default.seed == 123

def test_generate_collinear_design_shape():
    """Test the shape of the generated design."""
    n_points = 10
    poor = Poor(k=2)
    design = poor.generate_collinear_design(n_points)
    
    assert isinstance(design, np.ndarray)
    assert design.shape == (n_points, 2)

def test_generate_collinear_design_values():
    """Test the values of the generated design (check collinearity)."""
    n_points = 5
    poor = Poor(k=2)
    
    # Set global seed for reproducibility since Poor uses np.random.normal
    np.random.seed(42)
    
    design = poor.generate_collinear_design(n_points)
    
    # Check x_coords are linspace(0.1, 0.9, n_points)
    expected_x = np.linspace(0.1, 0.9, n_points)
    np.testing.assert_allclose(design[:, 0], expected_x)
    
    # Check y_coords match expected calculation with seed 42
    # y = 0.2 * x + 0.3 + noise (sigma=0.01)
    # We use the values captured from a run with seed 42
    expected_y = np.array([0.32496714, 0.35861736, 0.40647689, 0.4552303, 0.47765847])
    np.testing.assert_allclose(design[:, 1], expected_y, atol=1e-6)

def test_generate_collinear_design_custom_sigma():
    """Test generation with a custom sigma."""
    n_points = 5
    poor = Poor(k=2)
    
    # Set global seed for reproducibility
    np.random.seed(42)
    
    # sigma=0.1
    design = poor.generate_collinear_design(n_points, sigma=0.1)
    
    # Expected values for sigma=0.1 (previous default logic)
    # y = 0.2 * x + 0.3 + noise (sigma=0.1)
    expected_y = np.array([0.36967142, 0.34617357, 0.46476885, 0.59230299, 0.45658466])
    np.testing.assert_allclose(design[:, 1], expected_y, atol=1e-6)

def test_generate_collinear_design_invalid_k():
    """Test that ValueError is raised when k != 2."""
    poor = Poor(k=3)
    with pytest.raises(ValueError, match="Collinear design currently implemented for 2D only."):
        poor.generate_collinear_design(10)

def test_generate_collinear_design_zero_points():
    """Test generation with 0 points."""
    poor = Poor(k=2)
    design = poor.generate_collinear_design(0)
    assert design.shape == (0, 2)

def test_generate_collinear_design_one_point():
    """Test generation with 1 point."""
    poor = Poor(k=2)
    
    # Set global seed for reproducibility
    np.random.seed(42)
    
    design = poor.generate_collinear_design(1)
    assert design.shape == (1, 2)
    
    # x should be 0.1 (start of linspace)
    # y calculation: 0.2*0.1 + 0.3 + noise (sigma=0.01)
    assert design[0, 0] == 0.1
    expected_y = 0.32496714
    np.testing.assert_allclose(design[0, 1], expected_y, atol=1e-6)
