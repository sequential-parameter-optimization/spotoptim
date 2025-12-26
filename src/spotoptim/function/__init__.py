"""
Analytical test functions for optimization.
"""

from .so import rosenbrock, ackley, michalewicz, robot_arm_hard
from .mo import mo_conv2_min, fonseca_fleming, kursawe, mo_conv2_max
from .torch_objective import TorchObjective

__all__ = [
    "rosenbrock",
    "ackley",
    "michalewicz",
    "robot_arm_hard",
    "mo_conv2_min",
    "fonseca_fleming",
    "kursawe",
    "mo_conv2_max",
    "TorchObjective",
]
