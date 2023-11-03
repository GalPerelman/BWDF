import os
import datetime
import pytz
import pandas as pd

import utils

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES = os.path.join(ROOT_DIR, "resources")

DMA_NAMES = ["DMA A (L/s)", "DMA B (L/s)", "DMA C (L/s)", "DMA D (L/s)", "DMA E (L/s)", "DMA F (L/s)", "DMA G (L/s)",
             "DMA H (L/s)", "DMA I (L/s)", "DMA J (L/s)"]

EXOG_COLUMNS = ['is_dst', 'is_special', 'day', 'month', 'Rainfall depth (mm)', 'Air temperature (°C)',
                'Windspeed (km/h)', 'week_num', 'Air humidity (%)', 'weekday_int', 'hour']

WEATHER_COLUMNS = ['Rainfall depth (mm)', 'Air temperature (°C)', 'Windspeed (km/h)', 'Air humidity (%)']

DMA_DATA = pd.DataFrame(index=DMA_NAMES,
                        data={'type': ['Hospital district',
                                       'Residential-countryside',
                                       'Residential-countryside',
                                       'Residential-suburban-commercial',
                                       'Residential-city-centre-commercial',
                                       'offices-sport-suburban',
                                       'Residential-city-centre',
                                       'City-centre',
                                       'Commercial-industrial-port',
                                       'Commercial-industrial-port'
                                       ],
                              'type_code': [0, 1, 1, 2, 2, 3, 2, 4, 4, 4],
                              'n': [162, 531, 607, 2094, 7955, 1135, 3180, 2901, 425, 776]
                              })

TZ = pytz.timezone('CET')

SUMMER_TIME = [
    {'start': TZ.localize(datetime.datetime(2021, 3, 28)), 'end': TZ.localize(datetime.datetime(2021, 10, 30))},
    {'start': TZ.localize(datetime.datetime(2022, 3, 27)), 'end': TZ.localize(datetime.datetime(2022, 10, 29))},
    {'start': TZ.localize(datetime.datetime(2023, 3, 26)), 'end': TZ.localize(datetime.datetime(2023, 10, 30))}
]

SPECIAL_DATES = [
    TZ.localize(datetime.datetime(2021, 1, 1)),
    TZ.localize(datetime.datetime(2021, 1, 6)),
    TZ.localize(datetime.datetime(2021, 4, 4)),
    TZ.localize(datetime.datetime(2021, 4, 5)),
    TZ.localize(datetime.datetime(2021, 4, 25)),
    TZ.localize(datetime.datetime(2021, 5, 1)),
    TZ.localize(datetime.datetime(2021, 6, 2)),
    TZ.localize(datetime.datetime(2021, 8, 15)),
    TZ.localize(datetime.datetime(2021, 11, 1)),
    TZ.localize(datetime.datetime(2021, 11, 3)),
    TZ.localize(datetime.datetime(2021, 12, 8)),
    TZ.localize(datetime.datetime(2021, 12, 25)),
    TZ.localize(datetime.datetime(2021, 12, 26)),

    TZ.localize(datetime.datetime(2022, 1, 1)),
    TZ.localize(datetime.datetime(2022, 1, 6)),
    TZ.localize(datetime.datetime(2022, 4, 17)),
    TZ.localize(datetime.datetime(2022, 4, 18)),
    TZ.localize(datetime.datetime(2022, 4, 25)),
    TZ.localize(datetime.datetime(2022, 5, 1)),
    TZ.localize(datetime.datetime(2022, 6, 2)),
    TZ.localize(datetime.datetime(2022, 8, 15)),
    TZ.localize(datetime.datetime(2022, 11, 1)),
    TZ.localize(datetime.datetime(2022, 11, 3)),
    TZ.localize(datetime.datetime(2022, 12, 8)),
    TZ.localize(datetime.datetime(2022, 12, 25)),
    TZ.localize(datetime.datetime(2022, 12, 26)),

    TZ.localize(datetime.datetime(2023, 1, 1)),
    TZ.localize(datetime.datetime(2023, 1, 6)),
]

TEST_TIMES = {
    'w1': {'start_train': TZ.localize(datetime.datetime(2021, 1, 1, 0, 0)),
           'start_test': TZ.localize(datetime.datetime(2022, 7, 25, 0, 0)),
           'end_test': TZ.localize(datetime.datetime(2022, 8, 1, 0, 0))},

    'w2': {'start_train': TZ.localize(datetime.datetime(2021, 1, 1, 0, 0)),
           'start_test': TZ.localize(datetime.datetime(2022, 10, 31, 0, 0)),
           'end_test': TZ.localize(datetime.datetime(2022, 11, 6, 0, 0))},

    'w3': {'start_train': TZ.localize(datetime.datetime(2021, 1, 1, 0, 0)),
           'start_test': TZ.localize(datetime.datetime(2023, 1, 16, 0, 0)),
           'end_test': TZ.localize(datetime.datetime(2023, 1, 23, 0, 0))},

    'w4': {'start_train': TZ.localize(datetime.datetime(2021, 1, 1, 0, 0)),
           'start_test': TZ.localize(datetime.datetime(2023, 3, 6, 0, 0)),
           'end_test': TZ.localize(datetime.datetime(2023, 3, 13, 0, 0))}
}

DATES_TO_TEST_EXTREME_RAINFALL = {
    'start_train': TZ.localize(datetime.datetime(2021, 1, 1, 0, 0)),
    'start_test': TZ.localize(datetime.datetime(2021, 9, 17, 0, 0)),
    'end_test': TZ.localize(datetime.datetime(2021, 9, 18, 0, 0))
}

DATES_OF_LATEST_WEEK = {
    'start_train': TZ.localize(datetime.datetime(2021, 1, 1, 0, 0)),
    'start_test': TZ.localize(datetime.datetime(2022, 7, 18, 0, 0)),
    'end_test': TZ.localize(datetime.datetime(2022, 7, 25, 0, 0))
}
