
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from spotoptim import SpotOptim

# Define a simple multi-objective function
def mo_fun(X):
    obj1 = np.sum(X**2, axis=1)
    obj2 = np.sum((X-1)**2, axis=1)
    return np.column_stack([obj1, obj2])

# Add metrics attribute for name discovery
def mo_fun_with_names(X):
    return mo_fun(X)
mo_fun_with_names.metrics = ["Obj A", "Obj B"]

# Single objective function
def so_fun(X):
    return np.sum(X**2, axis=1)

@pytest.fixture
def mock_plt():
    with patch("matplotlib.pyplot.show") as mock_show, \
         patch("matplotlib.pyplot.figure") as mock_fig, \
         patch("matplotlib.pyplot.plot") as mock_plot, \
         patch("matplotlib.pyplot.scatter") as mock_scatter, \
         patch("matplotlib.pyplot.axvspan") as mock_axvspan:
        yield {
            "show": mock_show,
            "figure": mock_fig,
            "plot": mock_plot,
            "scatter": mock_scatter,
            "axvspan": mock_axvspan
        }

def test_plot_progress_single_objective(mock_plt):
    """Test standard single objective plotting."""
    opt = SpotOptim(fun=so_fun, bounds=[(-1, 1)]*2, n_initial=5, max_iter=10)
    opt.optimize()
    
    # Use default params
    opt.plot_progress(show=True)
    
    mock_plt["show"].assert_called_once()
    assert mock_plt["plot"].call_count >= 1 # At least sequential evaluations

def test_plot_progress_mo_no_names(mock_plt):
    """Test MO plotting without explicit names."""
    opt = SpotOptim(fun=mo_fun, bounds=[(-1, 1)]*2, n_initial=5, max_iter=10)
    opt.optimize()
    
    opt.plot_progress(mo=True, show=True)
    
    # We expect calls to plot for sequential evaluations + best + 2 objectives
    # The MO plots should use labels "Objective 1", "Objective 2"
    
    plot_calls = mock_plt["plot"].call_args_list
    labels = [call.kwargs.get('label') for call in plot_calls]
    
    assert "Objective 1" in labels
    assert "Objective 2" in labels
    
def test_plot_progress_mo_with_names(mock_plt):
    """Test MO plotting with discovered names."""
    opt = SpotOptim(fun=mo_fun_with_names, bounds=[(-1, 1)]*2, n_initial=5, max_iter=10)
    opt.optimize()
    
    opt.plot_progress(mo=True, show=True)
    
    plot_calls = mock_plt["plot"].call_args_list
    labels = [call.kwargs.get('label') for call in plot_calls]
    
    assert "Obj A" in labels
    assert "Obj B" in labels

def test_plot_progress_mo_single_objective_fallback(mock_plt):
    """Test asking for MO plot on single objective optimization."""
    opt = SpotOptim(fun=so_fun, bounds=[(-1, 1)]*2, n_initial=5, max_iter=10)
    opt.optimize()
    
    # Should not crash, but also shouldn't plot MO lines since y_mo is None
    opt.plot_progress(mo=True, show=True)
    
    plot_calls = mock_plt["plot"].call_args_list
    labels = [call.kwargs.get('label') for call in plot_calls]
    
    # Should contain standard labels but NOT "Objective 1" etc
    assert "Sequential evaluations" in labels
    assert not any("Objective" in str(l) for l in labels if l != "Objective Value") # Exclude ylabel default

def test_plot_progress_raises_without_data():
    """Test error when calling before optimize."""
    opt = SpotOptim(fun=so_fun, bounds=[(-1, 1)]*2)
    with pytest.raises(ValueError, match="No optimization data available"):
        opt.plot_progress()
