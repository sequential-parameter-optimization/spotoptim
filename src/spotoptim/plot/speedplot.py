import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from spotoptim.models.llamas import llamas_pi


def get_limits(sld, experiment, s=0.1, measurements="measurements_preprocessed") -> tuple:
    """
    Calculate extended limits for specified measurements within a given experiment.

    This function retrieves the measurements data for a specified experiment and calculates
    extended limits for the variables 'm_V [kg/s]' and 'Pi_tot_V [-]', based on an
    optional scaling factor s. The limits are expanded by multiplying the minimum by (1 - s)
    and the maximum by (1 + s).

    Args:
        sld (dict): SpeedlineData object
            The SpeedlineData object containing the experiment data.
        experiment (str):
            The name of the experiment from which to retrieve the measurements.
        s (float):
            The scaling factor for extending the limits. Default is 0.1 (10%).

    Returns:
        tuple
            A tuple containing four float values:
            - min_m_V: The extended minimum of 'm_V [kg/s]'.
            - max_m_V: The extended maximum of 'm_V [kg/s]'.
            - min_Pi_tot_V: The extended minimum of 'Pi_tot_V [-]'.
            - max_Pi_tot_V: The extended maximum of 'Pi_tot_V [-]'.

    Examples:
        >>> from spotoptim.data.speedline_data import SpeedlineData
        >>> from spotoptim.plot.speedline import get_limits
        >>> sld = SpeedlineData(...)  # Assume a properly initialized SpeedlineData object
        >>> get_limits(sld, "experiment_1", s=0.1)
        (0.89, 6.42, 2.20, 3.06)
    """
    df = sld.get_data(experiment, measurements)

    # Calculate the extended limits for 'm_V [kg/s]'
    min_m_V = (1 - s) * df["m_V [kg/s]"].min()
    max_m_V = (1 + s) * df["m_V [kg/s]"].max()

    # Calculate the extended limits for 'Pi_tot_V [-]'
    min_Pi_tot_V = (1 - s) * df["Pi_tot_V [-]"].min()
    max_Pi_tot_V = (1 + s) * df["Pi_tot_V [-]"].max()

    return min_m_V, max_m_V, min_Pi_tot_V, max_Pi_tot_V


def plot_speedline_fit(sl_data, data_dict_name, axs=None, show=True, colormap="viridis", fontsize=10, xmin=0.0, xmax=1.1, ymin=0.0, ymax=1.1, grid_visible=True):
    """
    Plot the speedline fit for the given data.

    Args:
        sl_data (dict): Contains the data for speedlines.
        data_dict_name (str): The name of the data dictionary.
        axs (matplotlib.axes.Axes, optional): Pre-existing axes for the plot. Default is None.
        show (bool): Whether to display the plot. Default is True.
        colormap (str): The name of the matplotlib colormap to use. Default is 'viridis'.
    """
    if axs is None:
        fig, axs = plt.subplots()

    # Extract calculation data
    measurements_preprocessed = sl_data.datadicts[data_dict_name]["measurements_preprocessed"]
    massflow = np.linspace(0, xmax, 500)

    # Prepare colormap
    cmap = get_cmap(colormap)
    num_lines = measurements_preprocessed["Speedclass [m/s]"].nunique()

    # Iterate over each speedline group
    for idx, (speedline, group) in enumerate(measurements_preprocessed.groupby("Speedclass [m/s]")):
        # Retrieve normalized flow and pressure data
        plot_m_v_red_normalized = group["m_V [kg/s]"]
        plot_pi_tot_v_normalized = group["Pi_tot_V [-]"]

        # Access fit coefficients for the speedline
        model_coeff_row = sl_data.datadicts[data_dict_name]["model_coeff"]
        coeffs = model_coeff_row.loc[model_coeff_row["Speedline"] == speedline, "coeff"].values[0]

        # Select color from colormap
        color = cmap(idx / num_lines)

        # Plot the original data points
        axs.scatter(plot_m_v_red_normalized, plot_pi_tot_v_normalized, label=f"speed: {speedline}", color=color)

        # Compute and plot fitted line
        y_val = llamas_pi(coeffs, massflow)
        axs.plot(massflow, y_val, linestyle="dashed", color=color)

    # Label axes and configure plot aesthetics
    axs.set_xlabel("Massflow [m_V]", fontsize=fontsize)
    axs.set_ylabel("Pressure [Pi_tot_V]", fontsize=fontsize)
    axs.set_xlim(xmin, xmax)
    axs.set_ylim(ymin, ymax)
    axs.set_title(data_dict_name, fontsize=fontsize)
    axs.tick_params(axis="both", which="major", labelsize=10)
    axs.grid(visible=grid_visible)

    # Place legend outside plot
    axs.legend(loc="lower left", bbox_to_anchor=(1.02, 0), fontsize=fontsize)

    # Show plot if requested
    if show:
        plt.show()
