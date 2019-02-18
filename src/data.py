from typing import Optional
from datetime import datetime
from src.constants import *


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
