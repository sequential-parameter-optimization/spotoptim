"""Sampling methods for design of experiments."""

from spotoptim.sampling.mm import (
    rlh,
    jd,
    mm,
    mmphi,
    mmsort,
    perturb,
    mmlhs,
    phisort,
    subset,
    mmphi_intensive,
    mmphi_intensive_update,
    propose_mmphi_intensive_minimizing_point,
    bestlh,
    plot_mmphi_vs_n_lhs,
)

__all__ = [
    "rlh",
    "jd",
    "mm",
    "mmphi",
    "mmsort",
    "perturb",
    "mmlhs",
    "phisort",
    "subset",
    "mmphi_intensive",
    "mmphi_intensive_update",
    "propose_mmphi_intensive_minimizing_point",
    "bestlh",
    "plot_mmphi_vs_n_lhs",
]
