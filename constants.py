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

TIME_COLUMNS = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos',
                'weeknum_sin', 'weeknum_cos', 'is_dst', 'is_special']

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

# Experiments
DATES_TO_TEST_EXTREME_RAINFALL = {
    'start_train': TZ.localize(datetime.datetime(2021, 1, 1, 0, 0)),
    'start_test': TZ.localize(datetime.datetime(2021, 9, 17, 0, 0)),
    'end_test': TZ.localize(datetime.datetime(2021, 9, 24, 0, 0))
}

DATES_TO_TEST_NOV_HOLIDAYS = {
    'start_train': TZ.localize(datetime.datetime(2021, 1, 1, 0, 0)),
    'start_test': TZ.localize(datetime.datetime(2021, 11, 1, 0, 0)),
    'end_test': TZ.localize(datetime.datetime(2021, 11, 8, 0, 0))
}

DATES_TO_TEST_MISSING_WEATHER = {
    'start_train': TZ.localize(datetime.datetime(2021, 1, 1, 0, 0)),
    'start_test': TZ.localize(datetime.datetime(2022, 6, 6, 0, 0)),
    'end_test': TZ.localize(datetime.datetime(2022, 6, 13, 0, 0))
}

W1_LATEST_WEEK = {
    'start_train': TZ.localize(datetime.datetime(2021, 1, 1, 0, 0)),
    'start_test': TZ.localize(datetime.datetime(2022, 7, 18, 0, 0)),
    'end_test': TZ.localize(datetime.datetime(2022, 7, 25, 0, 0))
}

W1_PREV_YEAR = {
    'start_train': TZ.localize(datetime.datetime(2021, 1, 1, 0, 0)),
    'start_test': TZ.localize(datetime.datetime(2021, 7, 25, 0, 0)),
    'end_test': TZ.localize(datetime.datetime(2021, 8, 1, 0, 0))
}

W2_LATEST_WEEK = {
    'start_train': TZ.localize(datetime.datetime(2021, 1, 1, 0, 0)),
    'start_test': TZ.localize(datetime.datetime(2022, 10, 24, 0, 0)),
    'end_test': TZ.localize(datetime.datetime(2022, 10, 31, 0, 0))
}

W2_LATEST_TWO_WEEKS = {
    'start_train': TZ.localize(datetime.datetime(2021, 1, 1, 0, 0)),
    'start_test': TZ.localize(datetime.datetime(2022, 10, 17, 0, 0)),
    'end_test': TZ.localize(datetime.datetime(2022, 10, 24, 0, 0))
}

W2_PREV_YEAR = {
    'start_train': TZ.localize(datetime.datetime(2021, 1, 1, 0, 0)),
    'start_test': TZ.localize(datetime.datetime(2021, 10, 31, 0, 0)),
    'end_test': TZ.localize(datetime.datetime(2021, 11, 7, 0, 0))
}

W3_PREV_YEAR = {
    'start_train': TZ.localize(datetime.datetime(2021, 1, 1, 0, 0)),
    'start_test': TZ.localize(datetime.datetime(2022, 1, 17, 0, 0)),
    'end_test': TZ.localize(datetime.datetime(2022, 1, 24, 0, 0))
}

W3_PREV_YEAR2 = {
    'start_train': TZ.localize(datetime.datetime(2021, 1, 1, 0, 0)),
    'start_test': TZ.localize(datetime.datetime(2022, 1, 10, 0, 0)),
    'end_test': TZ.localize(datetime.datetime(2022, 1, 17, 0, 0))
}

W3_LATEST_WEEK = {
    'start_train': TZ.localize(datetime.datetime(2021, 1, 1, 0, 0)),
    'start_test': TZ.localize(datetime.datetime(2023, 1, 9, 0, 0)),
    'end_test': TZ.localize(datetime.datetime(2023, 1, 16, 0, 0))
}

W3_LATEST_TWO_WEEKS = {
    'start_train': TZ.localize(datetime.datetime(2021, 1, 1, 0, 0)),
    'start_test': TZ.localize(datetime.datetime(2023, 1, 2, 0, 0)),
    'end_test': TZ.localize(datetime.datetime(2023, 1, 9, 0, 0))
}

W4_PREV_YEAR = {
    'start_train': TZ.localize(datetime.datetime(2021, 1, 1, 0, 0)),
    'start_test': TZ.localize(datetime.datetime(2022, 3, 7, 0, 0)),
    'end_test': TZ.localize(datetime.datetime(2022, 3, 14, 0, 0))
}

W4_LATEST_WEEK = {
    'start_train': TZ.localize(datetime.datetime(2021, 1, 1, 0, 0)),
    'start_test': TZ.localize(datetime.datetime(2023, 2, 27, 0, 0)),
    'end_test': TZ.localize(datetime.datetime(2023, 3, 6, 0, 0))
}

W4_LATEST_TWO_WEEKS = {
    'start_train': TZ.localize(datetime.datetime(2021, 1, 1, 0, 0)),
    'start_test': TZ.localize(datetime.datetime(2023, 2, 20, 0, 0)),
    'end_test': TZ.localize(datetime.datetime(2023, 2, 27, 0, 0))
}

TEMP = {
    'start_train': TZ.localize(datetime.datetime(2021, 1, 1, 0, 0)),
    'start_test': TZ.localize(datetime.datetime(2022, 12, 12, 0, 0)),
    'end_test': TZ.localize(datetime.datetime(2022, 12, 19, 0, 0))
}

EXPERIMENTS_DATES = {
    0: DATES_TO_TEST_EXTREME_RAINFALL,
    1: DATES_TO_TEST_NOV_HOLIDAYS,
    2: DATES_TO_TEST_MISSING_WEATHER,
    3: W1_LATEST_WEEK,
    4: W1_PREV_YEAR,
    5: W2_LATEST_WEEK,
    6: W2_LATEST_TWO_WEEKS,
    7: W2_PREV_YEAR,
    8: W3_PREV_YEAR,
    9: W3_PREV_YEAR2,
    10: W3_LATEST_WEEK,
    11: W3_LATEST_TWO_WEEKS,
    12: W4_PREV_YEAR,
    13: W4_LATEST_WEEK,
    14: W4_LATEST_TWO_WEEKS,
    99: TEMP
}
