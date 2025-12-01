import numpy as np
from spotoptim.mo.pareto import is_pareto_efficient

def test_is_pareto_efficient_minimize():
    costs = np.array([
        [1, 2],
        [2, 1],
        [3, 3],
        [0, 4],
        [4, 0],
    ])
    mask = is_pareto_efficient(costs, minimize=True)
    # Only [1,2], [2,1], [0,4], [4,0] are Pareto efficient
    expected = np.array([True, True, False, True, True])
    assert np.array_equal(mask, expected)

def test_is_pareto_efficient_maximize():
    costs = np.array([
        [1, 2],
        [2, 1],
        [3, 3],
        [0, 4],
        [4, 0],
    ])
    mask = is_pareto_efficient(costs, minimize=False)
    # Only [3,3], [0,4], [4,0] are Pareto efficient
    expected = np.array([False, False, True, True, True])
    assert np.array_equal(mask, expected)

def test_is_pareto_efficient_single_point():
    costs = np.array([[5, 5]])
    mask = is_pareto_efficient(costs)
    assert np.array_equal(mask, [True])

def test_is_pareto_efficient_identical_points():
    costs = np.array([[1, 1], [1, 1], [1, 1]])
    mask = is_pareto_efficient(costs)
    assert np.array_equal(mask, [True, False, False])

def test_is_pareto_efficient_empty():
    costs = np.empty((0, 2))
    mask = is_pareto_efficient(costs)
    assert mask.size == 0
