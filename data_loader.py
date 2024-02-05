import datetime
import os
import pandas as pd
import numpy as np
import pytz

import constants

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class Loader:
    def __init__(self, inflow_data_file, weather_data_file):
        self.inflow = self.load_excel(os.path.join(constants.RESOURCES, inflow_data_file))
        self.weather = self.load_excel(os.path.join(constants.RESOURCES, weather_data_file))

    @staticmethod
    def load_excel(file_path: str):
        """
        This function loads Excel file with the BWDF structure
        After loading the file the function convert the datetime index to be timezone aware
        First column is a datetime index
        The datetime format is DD/MM/YYYY HH:mm
        :return: pandas.DataFrame with datetime index
        """
        df = pd.read_excel(file_path, index_col=0)
        df.index = pd.to_datetime(df.index, format="%d/%m/%Y %H:%M")
        df.index = df.index.tz_localize(constants.TZ, ambiguous='infer')
        return df
