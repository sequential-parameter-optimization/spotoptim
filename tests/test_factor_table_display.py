# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import numpy as np
from spotoptim import SpotOptim

def obj(X):
    return np.sum(X[:, :1]**2, axis=1)

def test_factor_table_display_results(capsys):
    """
    Test that print_results displays '-' for lower/upper bounds of factor variables.
    """
    opt = SpotOptim(
        fun=obj,
        bounds=[(-1.0, 1.0), ("A", "B", "C")],
        var_name=["x1", "cat"],
        var_type=["float", "factor"],
        n_initial=3,
        max_iter=3,
        seed=42,
        verbose=False
    )
    opt.optimize()
    
    # Capture print_results output
    opt.print_results(precision=2)
    captured = capsys.readouterr()
    output = captured.out
    
    # Check that factor row contains '-' for lower and upper
    # tabulate output row for 'cat' should look something like:
    # |    cat | factor |         B |       - |       - |       A |           - |
    
    # Find the line with 'cat'
    lines = output.split('\n')
    cat_line = next((line for line in lines if "cat" in line), None)
    
    assert cat_line is not None, "Factor variable 'cat' not found in output table"
    assert " - " in cat_line, "Factor variable 'cat' should have '-' in columns"
    # To be more specific, we expect two '-' entries for lower and upper
    assert cat_line.count(" - ") >= 2, "Factor variable 'cat' should have '-' for both lower and upper bounds"

def test_factor_table_display_design(capsys):
    """
    Test that print_design_table displays '-' for lower/upper bounds of factor variables.
    """
    opt = SpotOptim(
        fun=obj,
        bounds=[(-1.0, 1.0), ("A", "B", "C")],
        var_name=["x1", "cat"],
        var_type=["float", "factor"],
        n_initial=3,
        max_iter=3,
        seed=42,
        verbose=False
    )
    
    # Capture print_design_table output
    opt.print_design_table(precision=2)
    captured = capsys.readouterr()
    output = captured.out
    
    # Find the line with 'cat'
    lines = output.split('\n')
    cat_line = next((line for line in lines if "cat" in line), None)
    
    assert cat_line is not None, "Factor variable 'cat' not found in design table"
    assert " - " in cat_line, "Factor variable 'cat' should have '-' in columns"
    # To be more specific, we expect two '-' entries for lower and upper
    assert cat_line.count(" - ") >= 2, "Factor variable 'cat' should have '-' for both lower and upper bounds"
