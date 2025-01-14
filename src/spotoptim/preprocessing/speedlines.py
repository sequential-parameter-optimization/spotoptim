import pandas as pd


def normalize(sl_data, data_dict_name) -> dict:
    """
    Normalize the data of the selected item text and store it in the data object.
    Normalize the data by dividing the values by the maximum value of the respective column, so
    the values are between 0 and 1.

    Args:
    sl_data: dict, a SpeedlineData object
    selected_item_text: str, the selected item text

    Returns:
    dict, a SpeedlineData object with the normalized data stored in the data object

    Examples:
        >>> from spotoptim.data.speedline_data import SpeedlineData
            from spotoptim.preprocessing.speedlines import normalize
            import pandas as pd
            experiment_1_data = {"measurements": pd.DataFrame({
                "Speedclass [m/s]": ['400.0', '400.0', '400.0', '400.0', '400.0', '400.0'],
                'm_V [kg/s]': [4.975694, 4.685357, 4.462651, 4.130313, 3.921199, 3.876735],
                'Pi_tot_V [-]': [2.449388, 2.605501, 2.665467, 2.73563, 2.772715, 2.7847]
                ]})
                }
            sld_1 = SpeedlineData("experiment_1", experiment_1_data)
            normalize(sld_1, "experiment_1")
            sld_1.get_data("experiment_1", "measurements_preprocessed")

    """
    measurements = sl_data.get_data(data_dict_name, "measurements")

    # ensure that the "Speedclass [m/s]" column is of type string
    measurements["Speedclass [m/s]"] = measurements["Speedclass [m/s]"].astype(str)

    # Extract the numerical value from the "Speedclass [m/s]" column by taking
    # the first element of the resulting list after the split.
    # 250 m/s -> 250
    u2 = measurements["Speedclass [m/s]"].apply(lambda x: float(x.split()[0]))
    m_V = measurements["m_V [kg/s]"]
    Pi_tot_V = measurements["Pi_tot_V [-]"]

    if "measurements_preprocessed" not in sl_data.get_data(data_dict_name):
        sl_data.datadicts[data_dict_name]["measurements_preprocessed"] = pd.DataFrame()

    measurements_preprocessed = sl_data.datadicts[data_dict_name]["measurements_preprocessed"]
    measurements_preprocessed["Speedclass [m/s]"] = measurements["Speedclass [m/s]"]
    measurements_preprocessed["m_V [kg/s]"] = m_V / m_V.max()
    measurements_preprocessed["Pi_tot_V [-]"] = Pi_tot_V / Pi_tot_V.max()
    measurements_preprocessed["u2_norm"] = u2 / u2.max()

    sl_data.datadicts[data_dict_name]["measurements_preprocessed"] = measurements_preprocessed

    return sl_data


def preprocess(sl_data, data_dict_name, method=None) -> dict:
    """
    Preprocess the data based on the specified method.

    Args:
    sl_data:
        dict, a SpeedlineData object
    data_dict_name:
        str, the name of the data dictionary
    method:
        str, the method to use for preprocessing. Default is None, i.e.,
        no preprocessing (identical to 'natural').

    Returns:
        SpeedlineData dict with updated normalized data
    """
    measurements = sl_data.get_data(data_dict_name, "measurements")

    if method == "normalize":
        # Call the normalize function to process the data
        sl_data = normalize(sl_data, data_dict_name)
    elif method is None or method == "natural":
        # Create a measurements_preprocessed DataFrame with the same data as measurements
        measurements_preprocessed = measurements.copy()

        if "measurements_preprocessed" not in sl_data.get_data(data_dict_name):
            sl_data.datadicts[data_dict_name]["measurements_preprocessed"] = pd.DataFrame()

        sl_data.datadicts[data_dict_name]["measurements_preprocessed"] = measurements_preprocessed

    return sl_data
