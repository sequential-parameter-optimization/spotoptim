import numpy as np
from pykenn.preprocess.speedlines import preprocess_data
from pykenn.utils.names import get_source_file_names


def get_data(
    n: int = 4,
    loc: float = 0.0,
    sd: float = 0.0,
    s_mult: float = 1,
    feature_list=["u2 red [m/s]", "PI tot V [-]", "V tot V red [m³/s]", "Bereich u2red", "Pumpgrenzpunkt", "source_file", "Eta sV [-]"],
) -> tuple:
    """Processes pump data and applies transformations with optional noise.

    This function selects a CSV file based on the index `n`, processes its data
    to extract specific features, applies transformations, and optionally adds
    normally distributed noise. The processed data is then saved as a new CSV file.

    Args:
        n (int, optional): The index of the source CSV file to process. Defaults to 4.
        loc (float, optional): The mean (`loc`) of the normal distribution used to add
            noise to certain columns. Defaults to 0.0.
        sd (float, optional): The standard deviation (`std`) of the normal distribution
            used to add noise to certain columns. Defaults to 0.0.
        s_mult (float, optional): A multiplier applied to the 'Speedclass [m/s]' column.
            Defaults to 1.
        feature_list (list, optional): A list of features to extract from the CSV file.
            Defaults to ["u2 red [m/s]", "PI tot V [-]", "V tot V red [m³/s]", "Bereich u2red", "Pumpgrenzpunkt", "source_file", "Eta sV [-]"].

    Returns:
        tuple: A tuple containing the processed DataFrame and the experiment name.

    Raises:
        IOError: If the CSV file cannot be read.
        IndexError: If the index `n` is out of bounds.

    Examples:
        >>> from spotoptim.data.io import get_data
        >>> df, experiment_name = get_data(n=4, loc=0.0, sd=0.0, s_mult=1)

    """
    try:
        csv_speedlines = get_source_file_names()[n]
    except IndexError:
        print(f"Invalid index: {n}. Please provide a valid index for the source file.")
        raise
    except Exception as e:
        print(f"An error occurred while fetching source file name: {e}")
        raise

    print(f"Selected csv-file(s): {csv_speedlines}")
    df = preprocess_data(csv_filename=csv_speedlines, min_entries=3, include_pumpgrenzpunkt=True, feature_list=feature_list)

    # Convert the dataframe to the required format
    df = df[["speed_class", "PI tot V [-]", "V tot V red [m³/s]"]]
    df.columns = ["Speedclass [m/s]", "Pi_tot_V [-]", "m_V [kg/s]"]

    # Multiply and convert the 'Speedclass [m/s]' column
    df["Speedclass [m/s]"] = df["Speedclass [m/s]"].astype(float) * s_mult
    df["Speedclass [m/s]"] = df["Speedclass [m/s]"].astype(str)

    # Add noise to 'Pi_tot_V [-]' and 'm_V [kg/s]' columns
    noise_pi_tot_v = np.random.normal(loc=loc, scale=sd, size=len(df))
    noise_m_v = np.random.normal(loc=loc, scale=sd, size=len(df))

    df["Pi_tot_V [-]"] += noise_pi_tot_v
    df["m_V [kg/s]"] += noise_m_v

    experiment_name = csv_speedlines
    if loc != 0.0 or sd != 0.0 or s_mult != 1:
        experiment_name = f"{experiment_name}_{loc}_{sd}_{s_mult}"

    # Save as CSV
    df.to_csv(f"{experiment_name}.csv", index=False)
    print(f"{experiment_name}.csv saved")

    return df, experiment_name
