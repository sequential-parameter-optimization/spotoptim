# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np


def safe_float(v):
    """Convert a value to float, returning np.nan for non-convertible values.

    Args:
        v: Value to convert. Can be numeric, string, None, or any type.

    Returns:
        float: The float value, or np.nan if conversion fails.
    """
    try:
        return float(v)
    except (ValueError, TypeError):
        return np.nan
