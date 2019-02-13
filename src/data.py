from typing import Optional
from datetime import datetime

DATE = 0
SEASON = DATE + 1
HOLIDAY = SEASON + 1
WORKING_DAY = HOLIDAY + 1
WEATHER = WORKING_DAY + 1
TEMP = WEATHER + 1
FELT_TEMP = TEMP + 1
HUMIDITY = FELT_TEMP + 1
WIND_SPEED = HUMIDITY + 1
CASUAL_CNT = WIND_SPEED + 1
REGISTERED_CNT = CASUAL_CNT + 1
TOTAL_CNT = REGISTERED_CNT + 1


class Data:

    is_train: bool

    date: datetime
    season: int
    holiday: bool
    working_day: bool
    weather: int
    temperature: float
    felt_temperature: float
    humidity: float
    wind_speed: float
    casual_cnt: Optional[int]
    registered_cnt: Optional[int]
    total_cnt: Optional[int]

    def __init__(self, is_train: bool, row: list):

        self.is_train = is_train

        self.date = datetime.strptime(row[DATE], '%Y-%m-%d %H:%M:%S')
        self.season = int(row[SEASON])
        self.holiday = bool(int(row[HOLIDAY]))
        self.working_day = bool(int(row[WORKING_DAY]))
        self.weather = int(row[WEATHER])
        self.temperature = float(row[TEMP])
        self.felt_temperature = float(row[FELT_TEMP])
        self.humidity = float(row[HUMIDITY])
        self.wind_speed = float(row[WIND_SPEED])

        self.casual_cnt = int(row[CASUAL_CNT]) if is_train else None
        self.registered_cnt = int(row[REGISTERED_CNT]) if is_train else None
        self.total_cnt = int(row[TOTAL_CNT]) if is_train else None
