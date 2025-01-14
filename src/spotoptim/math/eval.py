import numpy as np
import pandas as pd
from sklearn import metrics
from spotoptim.models.llamas import llamas_m, llamas_pi


def get_errors(beta, m_V, Pi_tot_V, metric_name="mean_squared_error") -> tuple:
    """
    Calculate the errors for predicting y = f(x) as well as x = f(y) using an sklearn metric.

    Args:
        beta (any): Parameters for the llamas_m and llamas_pi functions.
        m_V (numpy.ndarray): The measured values for m.
        Pi_tot_V (numpy.ndarray): The measured values for Pi_tot.
        metric_name (str, optional): The metric to use for calculation. Default is "mean_squared_error".

    Returns:
        tuple:
            First entry: the error for using m_V as input and predicting Pi_tot_V, i.e., y = f(x).
            Second entry: the error for using Pi_tot_V as input and predicting m_V, i.e., x = f(y).

    Raises:
        ValueError: If the specified metric is not found in sklearn.metrics.

    Example:
        >>> import numpy as np
            from spotoptim.fit.single_fit import get_errors
            beta = [0.5, 1.2, 0.1, 0.2, 0.1]
            m_V = np.array([1.0, 2.0, 3.0])
            Pi_tot_V = np.array([3.0, 2.0, 1.0])
            get_errors(beta, m_V, Pi_tot_V, metric_name="mean_squared_error")
            get_errors(beta, m_V, Pi_tot_V, metric_name="mean_absolute_error")
    """
    # Check if the specified metric exists in the metrics module
    if hasattr(metrics, metric_name):
        metric_function = getattr(metrics, metric_name)
    else:
        raise ValueError(f"Metric '{metric_name}' not found in sklearn.metrics")

    # Calculate the predicted values for Pi_tot_V given the parameters beta and m_V
    # i.e., y = f(x)
    Pi_tot_V_hat = llamas_pi(beta, m_V)
    # Calculate the error if Pi_tot_V is the true value and Pi_tot_V_hat is the predicted value,
    # i.e., the error for y = f(x)
    error_Pi_tot_V = float(metric_function(Pi_tot_V, Pi_tot_V_hat))

    # Calculate the predicted values for m_V given the parameters beta and Pi_tot_V,
    # i.e., x = f(y)
    m_V_hat = llamas_m(beta, Pi_tot_V)
    # Calculate the error if m_V is the true value and m_V_hat is the predicted value,
    # i.e., the error for x = f(y)
    error_m_V = float(metric_function(m_V, m_V_hat))

    return error_Pi_tot_V, error_m_V


def analyze_optimization_results(res, verbosity=0) -> tuple:
    """
    Analyze the optimization results from a given dictionary and return summary statistics.

    Args:
        res (dict):
            A dictionary containing optimization results with each entry having a 'fun' value.
        verbosity (int):
            The level of verbosity. Default is 0.

    Returns:
        tuple: Return two pd.DataFrames.
        * The first DataFrame contains the 'fun' values for each configuration.
        * The second DataFrame contains statistics of these 'fun' values: mean, standard deviation, median, min, max.

    Examples:
        >>> from spotoptim.fit.single_fit import analyze_optimization_results
        >>> res = {
        ...     '250 m/s': {
        ...         'message': 'Optimization terminated successfully',
        ...         'success': True,
        ...         'status': 0,
        ...         'fun': 0.00037942819453796984,
        ...         'x': [2.461e-01, 5.654e-01, 6.163e-01, 3.837e-01, 2.000e+00],
        ...         'nit': 44,
        ...         'jac': [6.613e-03, 9.735e-08, 6.185e-10, 1.513e-03, 1.176e-03],
        ...         'nfev': 70,
        ...         'njev': 44,
        ...     },
        ...     '400.0': {
        ...         'message': 'Optimization terminated successfully',
        ...         'success': True,
        ...         'status': 0,
        ...         'fun': 0.00012062512556007201,
        ...         'x': [5.844e-01, 1.021e+00, 1.030e+00, 8.014e-01, 2.000e+00],
        ...         'nit': 58,
        ...         'jac': [6.076e-04, 9.177e-09, 8.813e-10, 9.295e-10, 1.734e-04],
        ...         'nfev': 83,
        ...         'njev': 58,
        ...     },
        ... }
        >>> fun_df, stats_df = analyze_optimization_results(res)
    """

    # Extract 'fun' values and create a DataFrame
    fun_values = {key: result["fun"] for key, result in res.items()}
    fun_df = pd.DataFrame(fun_values.items(), columns=["config", "error"])

    # Convert fun values to a list for calculations
    fun_list = list(fun_values.values())

    # Calculate statistics
    mean_fun = np.mean(fun_list)
    std_fun = np.std(fun_list)
    median_fun = np.median(fun_list)
    min_fun = np.min(fun_list)
    max_fun = np.max(fun_list)

    if verbosity > 0:
        print(f"mean: {mean_fun}")
        print(f"s.d.: {std_fun}")
        print(f"median: {median_fun}")
        print(f"min: {min_fun}")
        print(f"max: {max_fun}")

    # Create a DataFrame for statistics
    stats_df = {"mean": [mean_fun], "s.d.": [std_fun], "median": [median_fun], "min": [min_fun], "max": [max_fun]}
    stats_df = pd.DataFrame(stats_df)
    return fun_df, stats_df


def evaluate_optimization_results(sl_data, data_dict_name, res, metric_name="mean_squared_error", verbosity=0) -> tuple:
    """
    Evaluate the optimization results from a given dictionary and return summary statistics.

    Args:
        sl_data (dict): The dictionary containing the data for the speedlines.
        data_dict_name (str): The name of the data dictionary.
        res (dict): A dictionary containing optimization results with each entry having a 'fun' value.
        metric_name (str): The name of the metric function to use. Default is 'mean_squared_error'.
        verbosity (int): The level of verbosity. Default is 0.

    Returns:
        tuple: Return three pd.DataFrames.
        * The first DataFrame contains the errors for each configuration.
        * The second DataFrame contains statistics of these errors for Pi_tot_V: mean, standard deviation, median, min, max.
        * The third DataFrame contains statistics of these errors for m_V: mean, standard deviation, median, min, max.

    Examples:
        >>> from spotoptim.fit.single_fit import evaluate_optimization_results
            from spotoptim.data.speedline_data import SpeedlineData
            from spotoptim.preprocessing.speedlines import preprocess
            import pandas as pd
            from spotoptim.fit.single_fit import single_fit, update_model_coeff
            from spotoptim.plot.speedplot import plot_speedline_fit, get_limits
            from spotoptim.fit.single_fit import analyze_optimization_results

            df = pd.DataFrame({
                'Speedclass [m/s]': ['251 m/s', '251 m/s', '251 m/s', '251 m/s', '251 m/s', '251 m/s','400.0', '400.0', '400.0', '400.0', '400.0', '400.0'],
                'm_V [kg/s]': [2.73843, 2.373404, 2.121546, 1.851186, 1.694762, 1.632984,4.975694, 4.685357, 4.462651, 4.130313, 3.921199, 5.876735],
                'Pi_tot_V [-]': [2.424709, 1.459302, 1.510399, 1.538109, 1.555507, 1.564646, 2.449388, 2.605501, 2.665467, 2.73563, 2.772715, 2.7847]
                })
            experiment_1_data = {"measurements": df}
            sld_1 = SpeedlineData("experiment_1", experiment_1_data)
            preprocessing_method="normalize"
            sld_1 = preprocess(sld_1, "experiment_1", method=preprocessing_method)
            res = single_fit(sld_1, "experiment_1", metric_name="ortho")
            update_model_coeff(sl_data=sld_1, data_dict_name="experiment_1", opt_results=res)
            evaluate_optimization_results(sld_1, "experiment_1", res, metric_name="max_error")
                ({'251 m/s': {'error_Pi_tot_V': 0.2997677533423275,'error_m_V': 0.22597077186582987},
                '400.0': {'error_Pi_tot_V': 0.17086724747527882,'error_m_V': 0.2655702473288445}},
                        mean     s.d.    median       min       max
                0  0.235318  0.06445  0.235318  0.170867  0.299768,
                        mean    s.d.    median       min      max
                0  0.245771  0.0198  0.245771  0.225971  0.26557)
    """
    # Extract 'x' values and create a dict
    x_values = {key: result["x"] for key, result in res.items()}

    measurements_preprocessed = sl_data.datadicts[data_dict_name]["measurements_preprocessed"]
    err = {}

    # Perform fit specialized for superelliptical datapoints on each speedline
    for speedline, group in measurements_preprocessed.groupby("Speedclass [m/s]"):
        m_V = np.array(group["m_V [kg/s]"])
        Pi_tot_V = np.array(group["Pi_tot_V [-]"])

        # Iterate over each beta value in the dictionary and compute errors
        for key, beta in x_values.items():
            if key == speedline:
                error_Pi_tot_V, error_m_V = get_errors(beta, m_V, Pi_tot_V, metric_name=metric_name)
                # print(f"Key: {key}, Error predicting Pi_tot_V: {error_Pi_tot_V},  Error predicting m_V: {error_m_V}")
                err[key] = {"error_Pi_tot_V": error_Pi_tot_V, "error_m_V": error_m_V}

    # Initialize lists to store errors separately
    error_Pi_tot_V_values = []
    error_m_V_values = []

    # Populate the lists
    for key, error_values in err.items():
        error_Pi_tot_V_values.append(error_values["error_Pi_tot_V"])
        error_m_V_values.append(error_values["error_m_V"])

    # Calculate and print statistics for error_Pi_tot_V
    mean_error_Pi_tot_V = np.mean(error_Pi_tot_V_values)
    sd_error_Pi_tot_V = np.std(error_Pi_tot_V_values)
    median_error_Pi_tot_V = np.median(error_Pi_tot_V_values)
    min_error_Pi_tot_V = np.min(error_Pi_tot_V_values)
    max_error_Pi_tot_V = np.max(error_Pi_tot_V_values)
    # Create a DataFrame for statistics
    stats_pi_df = {"mean": [mean_error_Pi_tot_V], "s.d.": [sd_error_Pi_tot_V], "median": [median_error_Pi_tot_V], "min": [min_error_Pi_tot_V], "max": [max_error_Pi_tot_V]}
    stats_pi_df = pd.DataFrame(stats_pi_df)

    # Calculate and print statistics for error_m_V
    mean_error_m_V = np.mean(error_m_V_values)
    sd_error_m_V = np.std(error_m_V_values)
    median_error_m_V = np.median(error_m_V_values)
    min_error_m_V = np.min(error_m_V_values)
    max_error_m_V = np.max(error_m_V_values)
    # Create a DataFrame for statistics
    stats_m_df = {"mean": [mean_error_m_V], "s.d.": [sd_error_m_V], "median": [median_error_m_V], "min": [min_error_m_V], "max": [max_error_m_V]}
    stats_m_df = pd.DataFrame(stats_m_df)

    if verbosity > 0:
        print("Statistics for error_Pi_tot_V:")
        print(f"Mean: {mean_error_Pi_tot_V:.4f}")
        print(f"Standard Deviation: {sd_error_Pi_tot_V:.4f}")
        print(f"Median: {median_error_Pi_tot_V:.4f}")
        print(f"Min: {min_error_Pi_tot_V:.4f}")
        print(f"Max: {max_error_Pi_tot_V:.4f}")
        print()
        print("Statistics for error_m_V:")
        print(f"Mean: {mean_error_m_V:.4f}")
        print(f"Standard Deviation: {sd_error_m_V:.4f}")
        print(f"Median: {median_error_m_V:.4f}")
        print(f"Min: {min_error_m_V:.4f}")
        print(f"Max: {max_error_m_V:.4f}")
    return err, stats_pi_df, stats_m_df
