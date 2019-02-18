import numpy as np
from typing import List
from matplotlib import pyplot

import src.data_repository
from src.data import Data
from src.constants import *


def plot_counts_vs_attribute(attr, title, xtick=None):
    pyplot.figure()
    pyplot.title(title)

    pyplot.scatter(attr, registered_cnts, c='g', s=5, label="registered")
    pyplot.scatter(attr, casual_cnts, c='r', s=5, label="casual")

    if xtick:
        pyplot.xticks(xtick)

    pyplot.legend()

    pyplot.savefig("./figs/{}.pdf".format(title))
    pyplot.close()


if __name__ == "__main__":

    # ----------------- read all data -----------------
    whole_data: List[Data] = src.data_repository.read_train_data(TRAIN_FILE_PATH)

    dates = np.array([d.date for d in whole_data])

    months = np.array([d.date.month for d in whole_data])
    days = np.array([d.date.weekday() for d in whole_data])
    hours = np.array([d.date.hour for d in whole_data])
    seasons = np.array([d.season for d in whole_data])
    holidays = np.array([d.holiday for d in whole_data])
    working_days = np.array([d.working_day for d in whole_data])
    weathers = np.array([d.weather for d in whole_data])
    temperatures = np.array([d.temperature for d in whole_data])
    felt_temps = np.array([d.felt_temperature for d in whole_data])
    wind_speeds = np.array([d.wind_speed for d in whole_data])

    casual_cnts = np.array([d.casual_cnt for d in whole_data])
    registered_cnts = np.array([d.registered_cnt for d in whole_data])

    # ----------------- visualize -----------------
    plot_counts_vs_attribute(months, "months", range(1, 13))
    plot_counts_vs_attribute(days, "days", range(0, 7))
    plot_counts_vs_attribute(hours, "hours", range(0, 24))
    plot_counts_vs_attribute(seasons, "seasons", range(1, 5))
    plot_counts_vs_attribute(holidays, "holidays", range(0, 2))
    plot_counts_vs_attribute(working_days, "working_days", range(0, 2))
    plot_counts_vs_attribute(weathers, "weathers", range(1, 5))
    plot_counts_vs_attribute(temperatures, "temperatures")
    plot_counts_vs_attribute(felt_temps, "felt_temps")
    plot_counts_vs_attribute(wind_speeds, "wind_speeds")