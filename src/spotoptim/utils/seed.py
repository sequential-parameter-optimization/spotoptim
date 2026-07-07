# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Global seeding for fully reproducible torch experiments.

Ported from the ``seed_everything`` helper of the schu25a compressor-map
study (``src/rnn/utils.py``): seeds Python, NumPy, and torch (CPU, CUDA, and
MPS) and switches cuDNN to deterministic mode, so that repeated runs on the
same device produce bit-identical results.

Requires the ``torch`` optional extra (``pip install 'spotoptim[torch]'``).
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed every random number generator relevant to a torch experiment.

    Seeds Python's ``random``, ``PYTHONHASHSEED``, NumPy, and torch (CPU and
    CUDA unconditionally, MPS when available), and sets
    ``torch.backends.cudnn.deterministic = True`` and
    ``torch.backends.cudnn.benchmark = False``.

    Args:
        seed (int): The seed value.

    Note:
        - cuDNN determinism and benchmark flags are process-global side
          effects; they may slow down convolutional workloads.
        - Unlike the schu25a original, the MPS generator is only seeded when
          the MPS backend is available, so the function also works on
          CPU-only Linux hosts.

    Examples:
        ```{python}
        import torch
        from spotoptim.utils.seed import seed_everything

        seed_everything(42)
        a = torch.rand(3)
        seed_everything(42)
        b = torch.rand(3)
        print(torch.equal(a, b))
        ```
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
