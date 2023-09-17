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
        self.data = pd.merge(self.inflow, self.weather, left_index=True, right_index=True, how="outer")
        self.datetime_categorize()

    def datetime_categorize(self):
        """
        Function to add categories based on the datetime index
        Month, weekday, hour, special dates
        """
        self.data['month'] = self.data.index.month
        self.data['day'] = self.data.index.day
        self.data['hour'] = self.data.index.hour
        self.data['weekday'] = self.data.index.day_name()
        self.data['weekday_int'] = (self.data.index.weekday + 1) % 7 + 1
        self.data['week_num'] = self.data.index.strftime('%U').astype(int) + 1

        self.data.index = self.data.index.tz_localize('UTC').tz_convert(constants.TZ)

        def is_dst(dt):
            return dt.dst() != pd.Timedelta(0)

        self.data['is_dst'] = self.data.index.map(is_dst)
        self.data['is_special'] = self.data.index.normalize().isin(constants.SPECIAL_DATES)

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
