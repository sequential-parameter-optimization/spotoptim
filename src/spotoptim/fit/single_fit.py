import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import approx_fprime
from spotoptim.models.llamas import llamas_pi
from spotoptim.models.llamas import llamas_m
from spotoptim.math.metrics import magnitude, sklearn_metric
from typing import Optional, Dict
from spotpython.fun.objectivefunctions import Analytical


def calculate_bounds(m_V, Pi_tot_V, a_m_zsl=0.75, b_m_zsl=1.1, a_m_chl=0.9, b_m_chl=1.25, a_pi_chl=0.75, b_pi_chl=1.25, a_pi_zsl=0.90, b_pi_zsl=1.25, a_cur=2, b_cur=15) -> list:
    """
    Calculate bounds for optimization based on provided arrays.

    Args:
        m_V (numpy.ndarray): Normalized mass flow array.
        Pi_tot_V (numpy.ndarray): Normalized pressure ratio array.
        a_m_zsl (float): Lower bound scaling factor for m_zsl. Default is 0.75.
        b_m_zsl (float): Upper bound scaling factor for m_zsl. Default is 1.1.
        a_m_chl (float): Lower bound scaling factor for m_chl. Default is 0.9.
        b_m_chl (float): Upper bound scaling factor for m_chl. Default is 1.25.
        a_pi_chl (float): Lower bound scaling factor for pi_chl. Default is 0.75.
        b_pi_chl (float): Upper bound scaling factor for pi_chl. Default is 1.25.
        a_pi_zsl (float): Lower bound scaling factor for pi_zsl. Default is 0.90.
        b_pi_zsl (float): Upper bound scaling factor for pi_zsl. Default is 1.25.
        a_cur (float): Lower bound for cur. Default is 2.
        b_cur (float): Upper bound for cur. Default is 15.

    Returns:
        list of tuple: Bounds for each parameter.
        
    Examples:
        >>> from spotoptim.fit.single_fit import calculate_bounds
            m_V = np.array([1., 2., 3.])
            Pi_tot_V = np.array([4., 5., 6.])
            calculate_bounds(m_V, Pi_tot_V)
    """
    m_zsl_lb, m_zsl_ub = a_m_zsl * np.min(m_V), b_m_zsl * np.min(m_V)
    m_chl_lb, m_chl_ub = a_m_chl * np.max(m_V), b_m_chl * np.max(m_V)
    pi_chl_lb, pi_chl_ub = a_pi_chl * np.min(Pi_tot_V), b_pi_chl * np.min(Pi_tot_V)
    pi_zsl_lb, pi_zsl_ub = a_pi_zsl * np.max(Pi_tot_V), b_pi_zsl * np.max(Pi_tot_V)
    cur_lb, cur_ub = a_cur, b_cur

    return [(m_zsl_lb, m_zsl_ub), (pi_zsl_lb, pi_zsl_ub), (m_chl_lb, m_chl_ub), (pi_chl_lb, pi_chl_ub), (cur_lb, cur_ub)]


def calculate_initial_guesses(m_V, Pi_tot_V, s=0.05, c_0=3) -> list:
    """
    Calculate initial guesses for optimization based on provided arrays.

    Args:
        m_V (numpy.ndarray): Normalized mass flow array.
        Pi_tot_V (numpy.ndarray): Normalized pressure ratio array.
        s (float): Scaling factor for initial guesses. Default is 0.05.
        c_0 (float): Initial guess for cur. Default is 3.

    Returns:
        list: Initial guesses for each parameter.
        
    Examples:
        >>> from spotoptim.fit.single_fit import calculate_initial_guesses
            m_V = np.array([1., 2., 3.])
            Pi_tot_V = np.array([4., 5., 6.])
            calculate_initial_guesses(m_V, Pi_tot_V)            
    """
    initial_m_zsl = (1 - s) * np.min(m_V)
    initial_m_chl = (1 + s) * np.max(m_V)
    initial_pi_chl = (1 - s) * np.min(Pi_tot_V)
    initial_pi_zsl = (1 + s) * np.max(Pi_tot_V)
    initial_cur = c_0
    return [initial_m_zsl, initial_pi_zsl, initial_m_chl, initial_pi_chl, initial_cur]


def update_model_coeff(sl_data, data_dict_name, opt_results):
    """
    Update the model_coeff DataFrame with optimization results.

    Args:
        sl_data (dict): The dictionary containing the data for the speedlines.
        data_dict_name (str): The name of the data dictionary.
        opt_results (dict): A dictionary mapping each speedline to its OptimizeResult object.
    """
    # Ensure model_coeff exists
    if "model_coeff" not in sl_data.datadicts[data_dict_name]:
        sl_data.datadicts[data_dict_name]["model_coeff"] = pd.DataFrame()

    # Get or create model_coeff
    model_coeff = sl_data.datadicts[data_dict_name]["model_coeff"]

    # Iterate over optimization results and update DataFrame
    for speedline, result in opt_results.items():
        # check if result is a dictionary
        if isinstance(result, dict):
            res_list = result["x"]
        # otherwise, result is a single OptimizeResult object
        else:
            res_list = result.x.tolist()
        new_row = pd.DataFrame(
            {
                "Speedline": [speedline],
                "coeff": [res_list],
            }
        )
        model_coeff = pd.concat([model_coeff, new_row], ignore_index=True)

    sl_data.datadicts[data_dict_name]["model_coeff"] = model_coeff
    return sl_data


def sum_squared(x, y) -> float:
    """
    Calculate the sum of squared elements of two arrays.
    Identical to the magnitude of the hypoteneuse of two arrays.

    Args:
        x (numpy.ndarray): The first array.
        y (numpy.ndarray): The second array.

    Returns:
        float: The sum of squared elements of the two arrays.

    Examples:
        >>> x = np.array([1., 2., 3.])
        >>> y = np.array([4., 5., 6.])
        >>> sum_squared(x, y)
        91.
        # Calculation: (1^2 + 2^2 + 3^2) + (4^2 + 5^2 + 6^2) = 14 + 77 = 91
    """
    s = np.square(x) + np.square(y)
    return np.sum(s)


def ortho(p1_x, p2_x, p1_y, p2_y, penalty=1e3) -> float:
    """
    Calculate the orthogonality penalty between two sets of points.
    NaN values in the differences are replaced with a specified penalty.
    Uses the magnitude function to calculate the squared magnitude of the hypoteneuse.

    Args:
        p1_x (numpy.ndarray): The x-coordinates of the first set of points.
        p2_x (numpy.ndarray): The x-coordinates of the second set of points.
        p1_y (numpy.ndarray): The y-coordinates of the first set of points.
        p2_y (numpy.ndarray): The y-coordinates of the second set of points.
        penalty (float, optional): The value to use for NaN differences. Default is 1e3.

    Returns:
        float: The calculated orthogonality penalty.

    Examples:
        >>> p1_x = np.array([1., 2., np.nan])
        >>> p2_x = np.array([1., 2., 3.])
        >>> p1_y = np.array([4., 5., np.nan])
        >>> p2_y = np.array([4., 5., 6.])
        >>> ortho(p1_x, p2_x, p1_y, p2_y)
        1000000.0
        # Calculation: Differences are (0, 0, NaN) for x and (0, 0, NaN) for y
        # Replace NaNs with penalty = 1000 giving differences (0, 0, 1000) for x and (0, 0, 1000) for y.
        # Magnitude calculation: (hypot(0, 0))^2 + (hypot(0, 0))^2 + (hypot(1000, 1000))^2 = 0 + 0 + 2000000 = 1000000
    """
    delta_x = p1_x - p2_x
    delta_y = p1_y - p2_y
    delta_x[np.isnan(delta_x)] = penalty
    delta_y[np.isnan(delta_y)] = penalty
    return magnitude(delta_x, delta_y)


class UserAnalytical(Analytical):
    def fun_user_function(self, X: np.ndarray, fun_control: Optional[Dict] = None) -> np.ndarray:
        """
        Custom new function: f(x) = x^4

        Args:
            X (np.ndarray): Input data as a 2D array.
            fun_control (Optional[Dict]): Control parameters for the function.

        Returns:
            np.ndarray: Computed values with optional noise.

        Examples:
            >>> import numpy as np
            >>> X = np.array([[1, 2, 3], [4, 5, 6]])
            >>> fun = UserAnalytical()
            >>> fun.fun_user_function(X)
        """
        X = self._prepare_input_data(X, fun_control)

        offset = np.ones(X.shape[1]) * self.offset
        y = np.sum((X - offset) ** 4, axis=1)

        # Add noise if specified in fun_control
        return self._add_noise(y)


def residual_func(beta, m_V, Pi_tot_V, metric_name="ortho", penalty=1e3) -> float:
    """
    Calculate the residual using either orthogonality or an sklearn metric.

    Args:
        beta (any): Parameters for the llamas_m and llamas_pi functions.
        m_V (numpy.ndarray): The measured values for m.
        Pi_tot_V (numpy.ndarray): The measured values for Pi_tot.
        metric_name (str, optional): The metric to use for calculation. Default is "ortho".
        penalty (float, optional): Penalty used in orthogonality calculation for NaN differences. Default is 1e3.

    Returns:
        float: The residual calculated by the specified metric.

    Examples:
        >>> from spotoptim.fit.single_fit import residual_func
            m_V = np.array([1., 2., 3.])
            Pi_tot_V = np.array([4., 5., 6.])
            beta = [1., 2., 3., 4., 5.]
            residual_func(beta, m_V, Pi_tot_V)
    """
    m_V_hat = llamas_m(beta, Pi_tot_V)
    Pi_tot_V_hat = llamas_pi(beta, m_V)
    if metric_name == "ortho":
        return ortho(p1_x=m_V, p1_y=Pi_tot_V, p2_x=m_V_hat, p2_y=Pi_tot_V_hat, penalty=penalty)
    else:
        return sklearn_metric(m_V, Pi_tot_V, m_V_hat, Pi_tot_V_hat, metric_name=metric_name)


def single_fit(
    sl_data,
    data_dict_name,
    method="CG",
    disp=False,
    metric_name="ortho",
    a_m_zsl=0.75,
    b_m_zsl=1.25,
    a_m_chl=0.75,
    b_m_chl=1.25,
    a_pi_chl=0.75,
    b_pi_chl=1.25,
    a_pi_zsl=0.75,
    b_pi_zsl=1.25,
    a_cur=2,
    b_cur=15,
    verbosity=0,
    **kwargs,
) -> dict:
    """
    Perform the optimization for each individual speedline in the dataset.
    The optimization is performed using a scipy.optimize minimize method.

    Args:
        sl_data (dict): The dictionary containing the data for the speedlines.
        data_dict_name (str): The name of the data dictionary.
        method (str): The optimization method to use. Default is 'CG'.
        metric_name (str): The name of the metric function to use. Default is 'ortho'.
        disp (bool): Whether to display the optimization results. Default is False.
        a_m_zsl (float): Lower bound scaling factor for m_zsl. Default is 0.75.
        b_m_zsl (float): Upper bound scaling factor for m_zsl. Default is 1.25.
        a_m_chl (float): Lower bound scaling factor for m_chl. Default is 0.75.
        b_m_chl (float): Upper bound scaling factor for m_chl. Default is 1.25.
        a_pi_chl (float): Lower bound scaling factor for pi_chl. Default is 0.75.
        b_pi_chl (float): Upper bound scaling factor for pi_chl. Default is 1.25.
        a_pi_zsl (float): Lower bound scaling factor for pi_zsl. Default is 0.75.
        b_pi_zsl (float): Upper bound scaling factor for pi_zsl. Default is 1.25.
        a_cur (float): Lower bound for cur. Default is 2.
        b_cur (float): Upper bound for cur. Default is 15.
        verbosity (int): The level of verbosity. Default is 0.
        **kwargs (any): Additional keyword arguments for the optimization method.

    Returns:
        dict: The updated dictionary containing the data for the speedlines.

    Examples:
        >>> from spotoptim.data.speedline_data import SpeedlineData
            from spotoptim.preprocessing.speedlines import normalize
            import pandas as pd
            from spotoptim.fit.single_fit import single_fit
            from spotoptim.plot.speedline import plot_speedline_fit
            experiment_1_data = {"measurements": pd.DataFrame({
                "Speedclass [m/s]": ['400.0', '400.0', '400.0', '400.0', '400.0', '400.0'],
                'm_V [kg/s]': [4.975694, 4.685357, 4.462651, 4.130313, 3.921199, 3.876735],
                'Pi_tot_V [-]': [2.449388, 2.605501, 2.665467, 2.73563, 2.772715, 2.7847]]})
                }
            experiment_2_data = {"measurements": pd.DataFrame({
                "Speedclass [m/s]": ['250 m/s', '250 m/s', '250 m/s', '250 m/s', '250 m/s', '250 m/s'],
                'm_V [kg/s]': [2.573843, 2.373404, 2.121546, 1.851186, 1.694762, 1.632984],
                'Pi_tot_V [-]': [1.424709, 1.459302, 1.510399, 1.538109, 1.555507, 1.564646]]})
                }
            sld_1 = SpeedlineData("experiment_1", experiment_1_data)
            normalize(sld_1, "experiment_1")
            # sld_1.get_data("experiment_1", "measurements_preprocessed")
            sld_1 = single_fit(sld_1, "experiment_1")
            # sld_1.get_data("experiment_1", "model_coeff")
            print(sld_1.get_data("experiment_1", "model_coeff")["coeff"].tolist())
            # plot_speedline_fit(sld_1, "experiment_1")
            sld_2 = SpeedlineData("experiment_2", experiment_2_data)
            normalize(sld_2, "experiment_2")
            sld_2 = single_fit(sld_2, "experiment_2")
            print(sld_2.get_data("experiment_2", "model_coeff")["coeff"].tolist())
                [[0.5843508965784472, 1.0207193924783673, 1.0299986631402998, 0.8014410568656544, 2.0]]
                [[0.47584021247605235, 1.0062932139273908, 1.1911584262681552, 0.6829223671041246, 2.0]]
        >>> from spotoptim.data.speedline_data import SpeedlineData
            from spotoptim.preprocessing.speedlines import preprocess
            import pandas as pd
            from spotoptim.fit.single_fit import single_fit
            experiment_name = "experiment_1"
            df = pd.DataFrame({
                'Speedclass [m/s]': ['250 m/s', '250 m/s', '250 m/s', '250 m/s', '250 m/s', '250 m/s','400.0', '400.0', '400.0', '400.0', '400.0', '400.0'],
                'm_V [kg/s]': [2.73843, 2.373404, 2.121546, 1.851186, 1.694762, 1.632984,4.975694, 4.685357, 4.462651, 4.130313, 3.921199, 3.876735],
                'Pi_tot_V [-]': [1.424709, 1.459302, 1.510399, 1.538109, 1.555507, 1.564646, 2.449388, 2.605501, 2.665467, 2.73563, 2.772715, 2.7847]
                })
            # Start the optimization fitting process
            experiment_1_data = {"measurements": df}
            sld_1 = SpeedlineData(experiment_name, experiment_1_data)
            # preprocessing_method="natural"
            preprocessing_method="normalize"
            sld_1 = preprocess(sld_1, experiment_name, method=preprocessing_method)
            single_fit(sld_1, experiment_name, method="SLSQP", metric_name="ortho")
                {'250 m/s':  message: Optimization terminated successfully
                success: True
                status: 0
                    fun: 0.001780768081950559
                        x: [ 2.461e-01  5.646e-01  6.412e-01  3.837e-01  2.000e+00]
                    nit: 48
                    jac: [ 1.601e-02  4.122e-06  2.151e-07  4.541e-03  3.165e-03]
                    nfev: 80
                    njev: 47,
                '400.0':  message: Optimization terminated successfully
                success: True
                status: 0
                    fun: 0.00012062512556003
                        x: [ 5.844e-01  1.021e+00  1.030e+00  8.014e-01  2.000e+00]
                    nit: 58
                    jac: [ 6.076e-04  1.127e-08  9.895e-10  1.106e-09  1.734e-04]
                    nfev: 83
                    njev: 58}
    """
    measurements_preprocessed = sl_data.datadicts[data_dict_name]["measurements_preprocessed"]

    # Dictionary to hold optimization results
    optimization_results = {}

    # Perform fit specialized for superelliptical datapoints on each speedline
    for speedline, group in measurements_preprocessed.groupby("Speedclass [m/s]"):
        m_V = np.array(group["m_V [kg/s]"])
        Pi_tot_V = np.array(group["Pi_tot_V [-]"])

        def gradient_func(beta, m_V, Pi_tot_V):
            return approx_fprime(beta, lambda beta: residual_func(beta, m_V, Pi_tot_V, metric_name=metric_name))

        # Perform optimization
        x0 = calculate_initial_guesses(m_V, Pi_tot_V)
        bounds = calculate_bounds(
            m_V, Pi_tot_V, a_m_zsl=a_m_zsl, b_m_zsl=b_m_zsl, a_m_chl=a_m_chl, b_m_chl=b_m_chl, a_pi_chl=a_pi_chl, b_pi_chl=b_pi_chl, a_pi_zsl=a_pi_zsl, b_pi_zsl=b_pi_zsl, a_cur=a_cur, b_cur=b_cur
        )
        result = minimize(fun=residual_func, x0=x0, args=(m_V, Pi_tot_V), method=method, jac=gradient_func, bounds=bounds, options={"maxiter": 10000, "ftol": 1e-15, "disp": disp}, tol=1e-15, **kwargs)

        # Save optimization result
        optimization_results[speedline] = result

    return optimization_results


def spot_fit(
    sl_data,
    data_dict_name,
    method="differential_evolution",
    disp=False,
    metric_name="ortho",
    a_m_zsl=0.75,
    b_m_zsl=1.25,
    a_m_chl=0.75,
    b_m_chl=1.25,
    a_pi_chl=0.75,
    b_pi_chl=1.25,
    a_pi_zsl=0.75,
    b_pi_zsl=1.25,
    a_cur=2,
    b_cur=15,
    verbosity=0,
    **kwargs,
) -> dict:
    """
    Perform the optimization for each individual speedline in the dataset.
    The optimization is performed using `spotpython` method.

    Args:
        sl_data (dict): The dictionary containing the data for the speedlines.
        data_dict_name (str): The name of the data dictionary.
        method (str): The optimization method to use. Default is 'differential_evolution'.
        metric_name (str): The name of the metric function to use. Default is 'ortho'.
        disp (bool): Whether to display the optimization results. Default is False.
        a_m_zsl (float): Lower bound scaling factor for m_zsl. Default is 0.75.
        b_m_zsl (float): Upper bound scaling factor for m_zsl. Default is 1.25.
        a_m_chl (float): Lower bound scaling factor for m_chl. Default is 0.75.
        b_m_chl (float): Upper bound scaling factor for m_chl. Default is 1.25.
        a_pi_chl (float): Lower bound scaling factor for pi_chl. Default is 0.75.
        b_pi_chl (float): Upper bound scaling factor for pi_chl. Default is 1.25.
        a_pi_zsl (float): Lower bound scaling factor for pi_zsl. Default is 0.75.
        b_pi_zsl (float): Upper bound scaling factor for pi_zsl. Default is 1.25.
        a_cur (float): Lower bound for cur. Default is 2.
        b_cur (float): Upper bound for cur. Default is 15.
        verbosity (int): The level of verbosity. Default is 0.
        **kwargs (any): Additional keyword arguments for the optimization method.

    Returns:
        dict: The updated dictionary containing the data for the speedlines.

    Examples:
        >>> from spotoptim.data.speedline_data import SpeedlineData
            from spotoptim.preprocessing.speedlines import normalize
            import pandas as pd
            from spotoptim.fit.single_fit import single_fit
            from spotoptim.plot.speedline import plot_speedline_fit
            experiment_1_data = {"measurements": pd.DataFrame({
                "Speedclass [m/s]": ['400.0', '400.0', '400.0', '400.0', '400.0', '400.0'],
                'm_V [kg/s]': [4.975694, 4.685357, 4.462651, 4.130313, 3.921199, 3.876735],
                'Pi_tot_V [-]': [2.449388, 2.605501, 2.665467, 2.73563, 2.772715, 2.7847]]})
                }
            experiment_2_data = {"measurements": pd.DataFrame({
                "Speedclass [m/s]": ['250 m/s', '250 m/s', '250 m/s', '250 m/s', '250 m/s', '250 m/s'],
                'm_V [kg/s]': [2.573843, 2.373404, 2.121546, 1.851186, 1.694762, 1.632984],
                'Pi_tot_V [-]': [1.424709, 1.459302, 1.510399, 1.538109, 1.555507, 1.564646]]})
                }
            sld_1 = SpeedlineData("experiment_1", experiment_1_data)
            normalize(sld_1, "experiment_1")
            # sld_1.get_data("experiment_1", "measurements_preprocessed")
            sld_1 = single_fit(sld_1, "experiment_1")
            # sld_1.get_data("experiment_1", "model_coeff")
            print(sld_1.get_data("experiment_1", "model_coeff")["coeff"].tolist())
            # plot_speedline_fit(sld_1, "experiment_1")
            sld_2 = SpeedlineData("experiment_2", experiment_2_data)
            normalize(sld_2, "experiment_2")
            sld_2 = single_fit(sld_2, "experiment_2")
            print(sld_2.get_data("experiment_2", "model_coeff")["coeff"].tolist())
                [[0.5843508965784472, 1.0207193924783673, 1.0299986631402998, 0.8014410568656544, 2.0]]
                [[0.47584021247605235, 1.0062932139273908, 1.1911584262681552, 0.6829223671041246, 2.0]]
        >>> from spotoptim.data.speedline_data import SpeedlineData
            from spotoptim.preprocessing.speedlines import preprocess
            import pandas as pd
            from spotoptim.fit.single_fit import single_fit
            experiment_name = "experiment_1"
            df = pd.DataFrame({
                'Speedclass [m/s]': ['250 m/s', '250 m/s', '250 m/s', '250 m/s', '250 m/s', '250 m/s','400.0', '400.0', '400.0', '400.0', '400.0', '400.0'],
                'm_V [kg/s]': [2.73843, 2.373404, 2.121546, 1.851186, 1.694762, 1.632984,4.975694, 4.685357, 4.462651, 4.130313, 3.921199, 3.876735],
                'Pi_tot_V [-]': [1.424709, 1.459302, 1.510399, 1.538109, 1.555507, 1.564646, 2.449388, 2.605501, 2.665467, 2.73563, 2.772715, 2.7847]
                })
            # Start the optimization fitting process
            experiment_1_data = {"measurements": df}
            sld_1 = SpeedlineData(experiment_name, experiment_1_data)
            # preprocessing_method="natural"
            preprocessing_method="normalize"
            sld_1 = preprocess(sld_1, experiment_name, method=preprocessing_method)
            single_fit(sld_1, experiment_name, method="SLSQP", metric_name="ortho")
                {'250 m/s':  message: Optimization terminated successfully
                success: True
                status: 0
                    fun: 0.001780768081950559
                        x: [ 2.461e-01  5.646e-01  6.412e-01  3.837e-01  2.000e+00]
                    nit: 48
                    jac: [ 1.601e-02  4.122e-06  2.151e-07  4.541e-03  3.165e-03]
                    nfev: 80
                    njev: 47,
                '400.0':  message: Optimization terminated successfully
                success: True
                status: 0
                    fun: 0.00012062512556003
                        x: [ 5.844e-01  1.021e+00  1.030e+00  8.014e-01  2.000e+00]
                    nit: 58
                    jac: [ 6.076e-04  1.127e-08  9.895e-10  1.106e-09  1.734e-04]
                    nfev: 83
                    njev: 58}
    """
    measurements_preprocessed = sl_data.datadicts[data_dict_name]["measurements_preprocessed"]

    # Dictionary to hold optimization results
    optimization_results = {}

    # Perform fit specialized for superelliptical datapoints on each speedline
    for speedline, group in measurements_preprocessed.groupby("Speedclass [m/s]"):
        m_V = np.array(group["m_V [kg/s]"])
        Pi_tot_V = np.array(group["Pi_tot_V [-]"])

        def gradient_func(beta, m_V, Pi_tot_V):
            return approx_fprime(beta, lambda beta: residual_func(beta, m_V, Pi_tot_V, metric_name=metric_name))

        # Perform optimization
        x0 = calculate_initial_guesses(m_V, Pi_tot_V)
        bounds = calculate_bounds(
            m_V, Pi_tot_V, a_m_zsl=a_m_zsl, b_m_zsl=b_m_zsl, a_m_chl=a_m_chl, b_m_chl=b_m_chl, a_pi_chl=a_pi_chl, b_pi_chl=b_pi_chl, a_pi_zsl=a_pi_zsl, b_pi_zsl=b_pi_zsl, a_cur=a_cur, b_cur=b_cur
        )
        result = minimize(fun=residual_func, x0=x0, args=(m_V, Pi_tot_V), method=method, jac=gradient_func, bounds=bounds, options={"maxiter": 10000, "ftol": 1e-15, "disp": disp}, tol=1e-15, **kwargs)

        # Save optimization result
        optimization_results[speedline] = result

    return optimization_results
