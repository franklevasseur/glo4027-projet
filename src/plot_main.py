import numpy as np
from typing import List
from matplotlib import pyplot

import data_repository
from data import Data
from constants import *


def plot_counts_vs_attribute(attr, title, xtitle, xtick=None):
    pyplot.figure()
    pyplot.xlabel(xtitle)
    pyplot.ylabel("Nombre de locations")

    pyplot.scatter(attr, registered_cnts, c='g', s=5, label="inscrits")
    pyplot.scatter(attr, casual_cnts, c='r', s=5, label="occasionnels")

    if xtick:
        pyplot.xticks(xtick)

    pyplot.legend()

    pyplot.savefig("./figs/{}.png".format(title))
    pyplot.close()


def plot_distribution(attr, title, xtitle):
    pyplot.figure()

    pyplot.xlabel(xtitle)
    pyplot.hist(attr, bins='auto')

    pyplot.savefig("./figs/hist_{}.png".format(title))
    pyplot.close()


if __name__ == "__main__":

    # ----------------- read all data -----------------
    whole_data: List[Data] = data_repository.read_train_data(TRAIN_FILE_PATH)

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
    total_counts = np.array([d.total_cnt for d in whole_data])

    # ----------------- counts vs dates ----------------------
    pyplot.figure()
    pyplot.xlabel("Date")
    pyplot.ylabel("Nombre de locations")
    pyplot.scatter(dates, total_counts, c='b', s=5)
    pyplot.savefig("./figs/totalcountVsdate.png")
    pyplot.close()

    pyplot.figure()
    pyplot.xlabel("Date")
    pyplot.ylabel("Nombre de locations")
    pyplot.scatter(dates, registered_cnts, c='g', s=5, label="usagers inscrits")
    pyplot.scatter(dates, casual_cnts, c='r', s=5, label="usagers occasionnels")

    pyplot.legend()
    pyplot.savefig("./figs/countsVsdate.png")
    pyplot.close()

    # ----------------- attributes vs counts -----------------
    plot_counts_vs_attribute(months, "months", "mois", range(1, 13))
    plot_counts_vs_attribute(days, "days", "jour", range(0, 7))
    plot_counts_vs_attribute(hours, "hours", "heure", range(0, 24))
    plot_counts_vs_attribute(seasons, "seasons", "saison", range(1, 5))
    plot_counts_vs_attribute(holidays, "holidays", "jour férié", range(0, 2))
    plot_counts_vs_attribute(working_days, "working_days", "jour ouvrable", range(0, 2))
    plot_counts_vs_attribute(weathers, "weathers", "météo", range(1, 5))
    plot_counts_vs_attribute(temperatures, "temperatures", "température")
    plot_counts_vs_attribute(felt_temps, "felt_temps", "température ressentie")
    plot_counts_vs_attribute(wind_speeds, "wind_speeds", "vitesse des vents")

    # ----------------- distributions -----------------
    plot_distribution(registered_cnts, "registered", "Nombre de locations d'usagers inscrits")
    plot_distribution(casual_cnts, "casual", "Nombre de locations d'usagers occasionnels")
    plot_distribution(temperatures, "temperatures", "temperatures")
    plot_distribution(felt_temps, "felt_temps", "temperatures ressenties")
    plot_distribution(wind_speeds, "wind_speeds", "vents")
