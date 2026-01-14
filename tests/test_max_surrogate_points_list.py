
import pytest
import numpy as np
from spotoptim import SpotOptim
from spotoptim.surrogate.kriging import Kriging

def test_max_surrogate_points_list_validation():
    """Test value error raised when lengths mismatch."""
    surrogates = [Kriging(), Kriging()]
    max_points = [10] # Mismatch length (2 surrogates)
    
    with pytest.raises(ValueError, match="Length of max_surrogate_points"):
        SpotOptim(
            fun=lambda x: x.sum(),
            bounds=[(0,1)],
            surrogate=surrogates,
            max_surrogate_points=max_points
        )

def test_max_surrogate_points_broadcast():
    """Test single int broadcast to list for multiple surrogates."""
    surrogates = [Kriging(), Kriging()]
    opt = SpotOptim(
        fun=lambda x: x.sum(),
        bounds=[(0,1)],
        surrogate=surrogates,
        max_surrogate_points=50
    )
    assert opt._max_surrogate_points_list == [50, 50]
    assert opt._active_max_surrogate_points == 50

def test_max_surrogate_points_switching():
    """Test that active max points parameter changes with surrogate switch."""
    surrogates = [Kriging(), Kriging()]
    max_points = [10, 20]
    # Force deterministic probabilities to control switching
    # E.g. [1.0, 0.0] then [0.0, 1.0] manually or by checking internal state logic
    
    opt = SpotOptim(
        fun=lambda x: x.sum(),
        bounds=[(0,1)],
        surrogate=surrogates,
        prob_surrogate=[0.5, 0.5],
        max_surrogate_points=max_points,
        seed=42
    )
    
    # We can't easily force switch without mocking RNG or running many iters.
    # Instead, we can manually trigger _fit_scheduler behavior logic or inspect internal list.
    assert opt._max_surrogate_points_list == [10, 20]
    
    # Initial state (first surrogate)
    assert opt.surrogate == surrogates[0]
    assert opt._active_max_surrogate_points == 10
    
    # Manually switch to second surrogate
    opt.surrogate = surrogates[1]
    # Update active manually (mimicking _fit_scheduler) to verify intended behavior would work
    # Or actually call _fit_scheduler with mocked probability?
    
    # Let's rely on _fit_scheduler property: it uses self.rng.
    # We can mock self.rng.choice
    
    class MockRNG:
        def choice(self, n, p=None):
            return 1 # Force index 1
            
    opt.rng = MockRNG()

    # Needs data to fit surrogate
    opt.X_ = np.array([[0.5]])
    opt.y_ = np.array([0.5])
    
    # Call _fit_scheduler
    opt._fit_scheduler()
    
    assert opt.surrogate == surrogates[1]
    assert opt._active_max_surrogate_points == 20

def test_single_surrogate_behavior():
    """Ensure standard single surrogate behavior preserved."""
    opt = SpotOptim(
        fun=lambda x: x.sum(),
        bounds=[(0,1)],
        max_surrogate_points=15
    )
    assert opt._active_max_surrogate_points == 15
    assert opt._max_surrogate_points_list is None
    
    # Fake fit call to check selection dispatcher wiring
    X = np.random.rand(20, 1)
    y = np.random.rand(20)
    
    # Mock _selection_dispatcher to verify it gets called
    called = False
    def mock_dispatcher(X, y):
        nonlocal called
        called = True
        return X[:15], y[:15]
        
    opt._selection_dispatcher = mock_dispatcher
    opt._fit_surrogate(X, y)
    
    assert called
