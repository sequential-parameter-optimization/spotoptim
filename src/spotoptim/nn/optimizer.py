# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Torch optimizer factory with per-optimizer base learning rates.

Ported from ``spotpython.hyperparameters.optimizer``. Instead of tuning one
learning rate across optimizers with very different scales, a single
``lr_mult`` hyperparameter multiplies each optimizer's individual base
learning rate (e.g. ``1.0`` for Adadelta, ``0.001`` for Adam). The constructor
arguments replicate spotPython exactly so that tuning results remain
comparable; in particular ``SparseAdam`` and ``RAdam`` ignore ``lr_mult``.

Requires the ``torch`` optional extra (``pip install 'spotoptim[torch]'``).
"""

from typing import Any, Iterable, Union

import torch
import torch.optim


def optimizer_handler(
    optimizer_name: str,
    params: Union[Iterable, list, torch.Tensor],
    lr_mult: float = 1.0,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """Return a torch optimizer with spotPython's base learning-rate convention.

    Args:
        optimizer_name (str): Name of the optimizer. Options:
            - "Adadelta": base lr 1.0 (times ``lr_mult``)
            - "Adagrad": base lr 0.01 (times ``lr_mult``)
            - "Adam": base lr 0.001 (times ``lr_mult``)
            - "AdamW": base lr 0.001 (times ``lr_mult``)
            - "SparseAdam": fixed lr 0.001 (``lr_mult`` ignored)
            - "Adamax": base lr 0.002 (times ``lr_mult``)
            - "ASGD": base lr 0.01 (times ``lr_mult``)
            - "LBFGS": base lr 1.0 (times ``lr_mult``)
            - "NAdam": base lr 0.002 (times ``lr_mult``)
            - "RAdam": fixed lr 0.001 (``lr_mult`` ignored)
            - "RMSprop": base lr 0.01 (times ``lr_mult``)
            - "Rprop": base lr 0.01 (times ``lr_mult``)
            - "SGD": base lr 0.001 (times ``lr_mult``)
        params (Union[Iterable, list, torch.Tensor]): The parameters to
            optimize, e.g. ``model.parameters()``.
        lr_mult (float, optional): Multiplier for the optimizer's base learning
            rate. Defaults to 1.0.
        **kwargs (Any): Ignored. Accepted for call-site compatibility.

    Returns:
        torch.optim.Optimizer: An instance of the specified optimizer.

    Raises:
        ValueError: If ``optimizer_name`` is not one of the supported names.

    Examples:
        ```{python}
        import torch
        from spotoptim.nn.optimizer import optimizer_handler

        weights = [torch.nn.Parameter(torch.zeros(3))]
        opt = optimizer_handler("Adam", weights, lr_mult=0.5)
        print(type(opt).__name__, opt.param_groups[0]["lr"])
        ```
    """
    if optimizer_name == "Adadelta":
        return torch.optim.Adadelta(
            params,
            lr=lr_mult * 1.0,
            rho=0.9,
            eps=1e-06,
            weight_decay=0,
            foreach=None,
            maximize=False,
        )
    elif optimizer_name == "Adagrad":
        return torch.optim.Adagrad(
            params,
            lr=lr_mult * 0.01,
            lr_decay=0,
            weight_decay=0,
            initial_accumulator_value=0,
            eps=1e-10,
            foreach=None,
            maximize=False,
        )
    elif optimizer_name == "Adam":
        return torch.optim.Adam(
            params,
            lr=lr_mult * 0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False,
            foreach=None,
            maximize=False,
            capturable=False,
            fused=None,
        )
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(
            params,
            lr=lr_mult * 0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.01,
            amsgrad=False,
            foreach=None,
            maximize=False,
            capturable=False,
        )
    elif optimizer_name == "SparseAdam":
        return torch.optim.SparseAdam(
            params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, maximize=False
        )
    elif optimizer_name == "Adamax":
        return torch.optim.Adamax(
            params,
            lr=lr_mult * 0.002,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            foreach=None,
            maximize=False,
        )
    elif optimizer_name == "ASGD":
        return torch.optim.ASGD(
            params,
            lr=lr_mult * 0.01,
            lambd=0.0001,
            alpha=0.75,
            t0=1000000.0,
            weight_decay=0,
            foreach=None,
            maximize=False,
        )
    elif optimizer_name == "LBFGS":
        return torch.optim.LBFGS(
            params,
            lr=lr_mult * 1,
            max_iter=20,
            max_eval=None,
            tolerance_grad=1e-07,
            tolerance_change=1e-09,
            history_size=100,
            line_search_fn=None,
        )
    elif optimizer_name == "NAdam":
        return torch.optim.NAdam(
            params,
            lr=lr_mult * 0.002,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            momentum_decay=0.004,
            foreach=None,
        )
    elif optimizer_name == "RAdam":
        return torch.optim.RAdam(
            params,
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            foreach=None,
        )
    elif optimizer_name == "RMSprop":
        return torch.optim.RMSprop(
            params,
            lr=lr_mult * 0.01,
            alpha=0.99,
            eps=1e-08,
            weight_decay=0,
            momentum=0,
            centered=False,
            foreach=None,
            maximize=False,
        )
    elif optimizer_name == "Rprop":
        return torch.optim.Rprop(
            params,
            lr=lr_mult * 0.01,
            etas=(0.5, 1.2),
            step_sizes=(1e-06, 50),
            foreach=None,
            maximize=False,
        )
    elif optimizer_name == "SGD":
        return torch.optim.SGD(
            params,
            lr=lr_mult * 1e-3,
            momentum=0,
            dampening=0,
            weight_decay=0,
            nesterov=False,
            maximize=False,
            foreach=None,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")
