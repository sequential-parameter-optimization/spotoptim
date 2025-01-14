from collections import defaultdict


class SpeedlineData:
    """
    SpeedlineData is a class that holds all the data for the speedline task.

    Attributes:
        datadicts: a dictionary of dictionaries that holds the data for each datadict.

    Examples:
        >>> from spotoptim.data.speedline_data import SpeedlineData
            import pandas as pd
            test_250 = {"measurements": pd.DataFrame({
                "Speedclass [m/s]": ['250 m/s', '250 m/s', '250 m/s', '250 m/s', '250 m/s', '250 m/s'],
                'm_V [kg/s]': [2.573843, 2.373404, 2.121546, 1.851186, 1.694762, 1.632984],
                'Pi_tot_V [-]': [1.424709, 1.459302, 1.510399, 1.538109, 1.555507, 1.564646]
                })
            }
            experiment_1 = {"measurements": pd.DataFrame({
                "Speedclass [m/s]": ['400.0', '400.0', '400.0', '400.0', '400.0', '400.0'],
                'm_V [kg/s]': [4.975694, 4.685357, 4.462651, 4.130313, 3.921199, 3.876735],
                'Pi_tot_V [-]': [2.449388, 2.605501, 2.665467, 2.73563, 2.772715, 2.7847]
                })
            }
            sld_1 = SpeedlineData("experiment_1", experiment_1)
            print(sld_1.datadicts["experiment_1"])
            sld_1.get_data("experiment_1", "measurements")
    """

    def __init__(self, datadict_name, data_dict):
        self.datadicts = {}
        if datadict_name is not None:
            self.datadicts[datadict_name] = data_dict
        else:
            self.datadicts = defaultdict(dict)

    def get_data(self, datadict_name, key=None) -> dict:
        """
        Get the data for the datadict name.

        Args:
        datadict_name: str, the name of the datadict
        key: str, the key of the data in the datadict

        Returns:
        dict, the data for the datadict name

        Examples:
            >>> from spotoptim.data.speedline_data import SpeedlineData
                import pandas as pd
                test_250 = {"measurements": pd.DataFrame({
                    "Speedclass [m/s]": ['250 m/s', '250 m/s', '250 m/s', '250 m/s', '250 m/s', '250 m/s'],
                    'm_V [kg/s]': [2.573843, 2.373404, 2.121546, 1.851186, 1.694762, 1.632984],
                    'Pi_tot_V [-]': [1.424709, 1.459302, 1.510399, 1.538109, 1.555507, 1.564646]
                    })
                }
                experiment_1 = {"measurements": pd.DataFrame({
                    "Speedclass [m/s]": ['400.0', '400.0', '400.0', '400.0', '400.0', '400.0'],
                    'm_V [kg/s]': [4.975694, 4.685357, 4.462651, 4.130313, 3.921199, 3.876735],
                    'Pi_tot_V [-]': [2.449388, 2.605501, 2.665467, 2.73563, 2.772715, 2.7847]
                    })
                }
                sld_1 = SpeedlineData("experiment_1", experiment_1)
                sld_1.get_data("experiment_1", "measurements")

        """
        if key is not None:
            return self.datadicts[datadict_name][key]
        else:
            return self.datadicts[datadict_name]
