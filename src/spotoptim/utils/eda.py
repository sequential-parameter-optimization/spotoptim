from tabulate import tabulate

# Try to import from spotpython, otherwise mock or re-implement necessary parts
try:
    from spotpython.hyperparameters.values import (
        get_default_values,
        get_bound_values,
        get_var_name,
        get_var_type,
        get_transform,
    )
except ImportError:
    # Minimal fallbacks if spotpython is not available (though it should be)
    pass


def get_stars(input_list: list) -> list:
    """Converts a list of values to a list of stars, which can be used to
        visualize the importance of a variable.

    Args:
        input_list (list): A list of values.

    Returns:
        (list):
            A list of strings.
    """
    output_list = []
    for value in input_list:
        if value > 99:
            output_list.append("***")
        elif value > 75:
            output_list.append("**")
        elif value > 50:
            output_list.append("*")
        elif value > 10:
            output_list.append(".")
        else:
            output_list.append("")
    return output_list


def print_exp_table(fun_control: dict, tablefmt="github", print_tab=True) -> str:
    """Generates a table with the design variables and their bounds.

    Args:
        fun_control (dict):
            A dictionary with function design variables.
        tablefmt (str):
            The format of the table. Defaults to "github".
        print_tab (bool):
            If True, the table is printed. Otherwise, the result code from tabulate
            is returned. Defaults to True.

    Returns:
        (str):
            a table with the design variables, their default values, and their bounds.
    """
    default_values = get_default_values(fun_control)
    defaults = list(default_values.values())
    tab = tabulate(
        {
            "name": get_var_name(fun_control),
            "type": get_var_type(fun_control),
            "default": defaults,
            "lower": get_bound_values(fun_control, "lower", as_list=True),
            "upper": get_bound_values(fun_control, "upper", as_list=True),
            "transform": get_transform(fun_control),
        },
        headers="keys",
        tablefmt=tablefmt,
    )
    if print_tab:
        print(tab)
    else:
        return tab


def print_res_table(
    spot: object = None, fun_control: dict = None, tablefmt="github", print_tab=True
) -> str:
    """
    Generates a table with the design variables and their bounds,
    after the run was completed.

    Args:
        spot (object):
            A SpotOptim object. Defaults to None.
        fun_control (dict):
             The fun_control dictionary.
        tablefmt (str):
            The format of the table. Defaults to "github".
        print_tab (bool):
            If True, the table is printed. Otherwise, the result code from tabulate
            is returned. Defaults to True.
    """
    # Use fun_control from SpotOptim if available (it might not be stored there directly)
    # But usually provided by user.
    if fun_control is None:
        # Try to guess or fallback
        raise ValueError("fun_control must be provided for print_res_table")

    default_values = get_default_values(fun_control)
    defaults = list(default_values.values())

    # In SpotOptim, the best result is min_X (transformed or original?)
    # SpotOptim stores min_X as the best X found.
    # We want to show the 'tuned' values.

    # Check if SpotOptim has results
    if spot.y_ is None or len(spot.y_) == 0:
        print("No results in SpotOptim object.")
        return

    # Best configuration found
    best_config_X = spot.min_X
    # If SpotOptim works with transformed values, we might need to inverse transform?
    # SpotOptim.min_X is usually in the space where we can interpret it if we know the encoding.
    # However, SpotOptim's X_ is stored in ORIGINAL scale (inverse transformed).
    # So min_X should also be in original scale?
    # Let's check SpotOptim.update_stats: self.min_X = self.X_[np.argmin(self.y_)]
    # And self.X_ is stored via _update_storage which calls _inverse_transform_X.
    # So yes, min_X is in original scale.

    tuned = best_config_X

    # Importance - SpotOptim doesn't have built-in importance yet.
    # Placeholder for now.
    importance = [0.0] * len(tuned)
    stars = get_stars(importance)

    # We need to make sure 'tuned' aligns with 'var_name' in fun_control
    # SpotOptim has var_name.
    var_names = get_var_name(fun_control)

    # If SpotOptim has varying variables (dimension reduction), we need to careful map.
    # But for now assume full match.

    tab = tabulate(
        {
            "name": var_names,
            "type": get_var_type(fun_control),
            "default": defaults,
            "lower": get_bound_values(fun_control, "lower", as_list=True),
            "upper": get_bound_values(fun_control, "upper", as_list=True),
            "tuned": tuned,
            "transform": get_transform(fun_control),
            "importance": importance,
            "stars": stars,
        },
        headers="keys",
        numalign="right",
        floatfmt=("", "", "", "", "", "", "", ".2f"),
        tablefmt=tablefmt,
    )
    if print_tab:
        print(tab)
    else:
        return tab


def gen_design_table(fun_control: dict, spot: object = None, tablefmt="github") -> str:
    # Simple wrapper or alias to print_exp_table / print_res_table logic
    if spot is None:
        return print_exp_table(fun_control, tablefmt, print_tab=False)
    else:
        return print_res_table(spot, fun_control, tablefmt, print_tab=False)
