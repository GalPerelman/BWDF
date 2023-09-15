import os
import pandas as pd

import constants


class Loader:
    def __init__(self):
        self.inflow = self.load_excel(os.path.join(constants.RESOURCES, "InflowData_1.xlsx"))
        self.weather = self.load_excel(os.path.join(constants.RESOURCES, "WeatherData_1.xlsx"))
        self.data = pd.merge(self.inflow, self.weather, left_index=True, right_index=True, how="outer")

    @staticmethod
    def load_excel(file_path: str):
        """
        This function loads Excel files with the BWDF structure
        First column is a datetime index
        The datetime format is DD/MM/YYYY HH:mm
        :return: pandas.DataFrame with datetime index
        """
        df = pd.read_excel(file_path, index_col=0)
        df.index = pd.to_datetime(df.index, format="%d/%m/%Y %H:%M")
        return df