import datetime
import os
import pandas as pd
import numpy as np
import pytz

import constants

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class Loader:
    def __init__(self):
        self.inflow = self.load_excel(os.path.join(constants.RESOURCES, "InflowData_1.xlsx"))
        self.weather = self.load_excel(os.path.join(constants.RESOURCES, "WeatherData_1.xlsx"))

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
        df.index = df.index.tz_localize('UTC').tz_convert(constants.TZ)
        return df
