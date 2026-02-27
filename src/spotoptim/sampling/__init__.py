# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Sampling methods for design of experiments."""

from spotoptim.sampling.mm import (
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
    mm_improvement_contour,
)

from spotoptim.sampling.lhs import rlh

from spotoptim.sampling.design import (
    generate_uniform_design,
    generate_collinear_design,
    generate_clustered_design,
    generate_sobol_design,
    generate_qmc_lhs_design,
    generate_grid_design,
    fullfactorial,
)

from spotoptim.sampling.effects import (
    randorient,
    screeningplan,
    screening_print,
    screening_plot,
    plot_all_partial_dependence,
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
    "mm_improvement_contour",
    "generate_uniform_design",
    "generate_collinear_design",
    "generate_clustered_design",
    "generate_sobol_design",
    "generate_qmc_lhs_design",
    "generate_grid_design",
    "fullfactorial",
    "randorient",
    "screeningplan",
    "screening_print",
    "screening_plot",
    "plot_all_partial_dependence",
]
