"""
Analytical test functions for optimization.
"""

from .so import rosenbrock, ackley, michalewicz, robot_arm_hard
from .mo import (
    activity_pred,
    conversion_pred,
    dtlz1,
    dtlz2,
    fonseca_fleming,
    fun_myer16a,
    kursawe,
    mo_conv2_max,
    mo_conv2_min,
    schaffer_n1,
    zdt1,
    zdt2,
    zdt3,
    zdt4,
    zdt6,
)
from .forr08a import aerofoilcd, branin, onevar
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
    "zdt1",
    "zdt2",
    "zdt3",
    "zdt4",
    "zdt6",
    "dtlz1",
    "dtlz2",
    "schaffer_n1",
    "conversion_pred",
    "activity_pred",
    "fun_myer16a",
    "aerofoilcd",
    "branin",
    "onevar",
    "TorchObjective",
]
