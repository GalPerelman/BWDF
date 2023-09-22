import os
import datetime

import pytz

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES = os.path.join(ROOT_DIR, "resources")

DMA_NAMES = ["DMA A (L/s)", "DMA B (L/s)", "DMA C (L/s)", "DMA D (L/s)", "DMA E (L/s)", "DMA F (L/s)", "DMA G (L/s)",
             "DMA H (L/s)", "DMA I (L/s)", "DMA J (L/s)"]

TZ = pytz.timezone('CET')

SUMMER_TIME = [
    {'start': datetime.datetime(2021, 3, 28, tzinfo=TZ), 'end': datetime.datetime(2021, 10, 30, tzinfo=TZ)},
    {'start': datetime.datetime(2022, 3, 27, tzinfo=TZ), 'end': datetime.datetime(2022, 10, 29, tzinfo=TZ)},
    {'start': datetime.datetime(2023, 3, 26, tzinfo=TZ), 'end': datetime.datetime(2023, 10, 30, tzinfo=TZ)}
]

SPECIAL_DATES = [
    datetime.datetime(2021, 1, 1, tzinfo=TZ),
    datetime.datetime(2021, 1, 6, tzinfo=TZ),
    datetime.datetime(2021, 4, 4, tzinfo=TZ),
    datetime.datetime(2021, 4, 5, tzinfo=TZ),
    datetime.datetime(2021, 4, 25, tzinfo=TZ),
    datetime.datetime(2021, 5, 1, tzinfo=TZ),
    datetime.datetime(2021, 6, 2, tzinfo=TZ),
    datetime.datetime(2021, 8, 15, tzinfo=TZ),
    datetime.datetime(2021, 11, 1, tzinfo=TZ),
    datetime.datetime(2021, 11, 3, tzinfo=TZ),
    datetime.datetime(2021, 12, 8, tzinfo=TZ),
    datetime.datetime(2021, 12, 25, tzinfo=TZ),
    datetime.datetime(2021, 12, 26, tzinfo=TZ),

    datetime.datetime(2022, 1, 1, tzinfo=TZ),
    datetime.datetime(2022, 1, 6, tzinfo=TZ),
    datetime.datetime(2022, 4, 17, tzinfo=TZ),
    datetime.datetime(2022, 4, 18, tzinfo=TZ),
    datetime.datetime(2022, 4, 25, tzinfo=TZ),
    datetime.datetime(2022, 5, 1, tzinfo=TZ),
    datetime.datetime(2022, 6, 2, tzinfo=TZ),
    datetime.datetime(2022, 8, 15, tzinfo=TZ),
    datetime.datetime(2022, 11, 1, tzinfo=TZ),
    datetime.datetime(2022, 11, 3, tzinfo=TZ),
    datetime.datetime(2022, 12, 8, tzinfo=TZ),
    datetime.datetime(2022, 12, 25, tzinfo=TZ),
    datetime.datetime(2022, 12, 26, tzinfo=TZ),

    datetime.datetime(2023, 1, 1, tzinfo=TZ),
    datetime.datetime(2023, 1, 6, tzinfo=TZ),
]

TEST_TIMES = {
    'w1': {'start': datetime.datetime(2022, 7, 25, tzinfo=TZ), 'end': datetime.datetime(2022, 8, 1, tzinfo=TZ)},
    'w2': {'start': datetime.datetime(2022, 10, 31, tzinfo=TZ), 'end': datetime.datetime(2022, 11, 6, tzinfo=TZ)},
    'w3': {'start': datetime.datetime(2023, 1, 16, tzinfo=TZ), 'end': datetime.datetime(2023, 1, 23, tzinfo=TZ)},
    'w4': {'start': datetime.datetime(2023, 3, 6, tzinfo=TZ), 'end': datetime.datetime(2023, 3, 13, tzinfo=TZ)}
}
