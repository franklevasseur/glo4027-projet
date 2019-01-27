from typing import Optional
from datetime import datetime

DAY_INDEX = 0
DAY_DATE = 1
DAY_SEASON = 2
DAY_YEAR = 3
DAY_MONTH = 4
DAY_HOLIDAY = 5
DAY_WEEKDAY = 6
DAY_WORKING_DAY = 7
DAY_WEATHER = 8
DAY_TEMP = 9
DAY_FELT_TEMP = 10
DAY_HUMIDITY = 11
DAY_WIND_SPEED = 12
DAY_CASUAL_CNT = 13
DAY_REGISTERED_CNT = 14
DAY_TOTAL_CNT = 15

HOUR_INDEX = 0
HOUR_DATE = 1
HOUR_SEASON = 2
HOUR_YEAR = 3
HOUR_MONTH = 4
HOUR_HOUR = 5
HOUR_HOLIDAY = 6
HOUR_WEEKDAY = 7
HOUR_WORKING_DAY = 8
HOUR_WEATHER = 9
HOUR_TEMP = 10
HOUR_FELT_TEMP = 11
HOUR_HUMIDITY = 12
HOUR_WIND_SPEED = 13
HOUR_CASUAL_CNT = 14
HOUR_REGISTERED_CNT = 15
HOUR_TOTAL_CNT = 16


class Data:

    is_day: bool

    index: int
    date: datetime
    season: int
    year: int
    month: int
    hour: Optional[int]
    holiday: bool
    weekday: int
    working_day: bool
    weather: int
    temperature: float
    felt_temperature: float
    humidity: float
    wind_speed: float
    casual_cnt: int
    registered_cnt: int

    def __init__(self, is_day_data: bool, row: list):

        self.is_day = is_day_data

        extract_date = lambda date_string: datetime.strptime(date_string, '%Y-%m-%d')

        if is_day_data:
            self.date = extract_date(row[DAY_DATE])
            self.season = int(row[DAY_SEASON])
            self.year = int(row[DAY_YEAR])
            self.month = int(row[DAY_MONTH])
            self.hour = None
            self.holiday = bool(row[DAY_HOLIDAY])
            self.weekday = int(row[DAY_WEEKDAY])
            self.working_day = bool(row[DAY_WORKING_DAY])
            self.weather = int(row[DAY_WEATHER])
            self.temperature = float(row[DAY_TEMP])
            self.felt_temperature = float(row[DAY_FELT_TEMP])
            self.humidity = float(row[DAY_HUMIDITY])
            self.wind_speed = float(row[DAY_WIND_SPEED])
            self.casual_cnt = int(row[DAY_CASUAL_CNT])
            self.registered_cnt = int(row[DAY_REGISTERED_CNT])
        else:
            self.date = extract_date(row[HOUR_DATE])
            self.season = int(row[HOUR_SEASON])
            self.year = int(row[HOUR_YEAR])
            self.month = int(row[HOUR_MONTH])
            self.hour = int(row[HOUR_HOUR])
            self.holiday = bool(row[HOUR_HOLIDAY])
            self.weekday = int(row[HOUR_WEEKDAY])
            self.working_day = bool(row[HOUR_WORKING_DAY])
            self.weather = int(row[HOUR_WEATHER])
            self.temperature = float(row[HOUR_TEMP])
            self.felt_temperature = float(row[HOUR_FELT_TEMP])
            self.humidity = float(row[HOUR_HUMIDITY])
            self.wind_speed = float(row[HOUR_WIND_SPEED])
            self.casual_cnt = int(row[HOUR_CASUAL_CNT])
            self.registered_cnt = int(row[HOUR_REGISTERED_CNT])