import numpy as np
from typing import List
from matplotlib import pyplot

import data_repository
from data import Data
from constants import *

from tabulate import tabulate

if __name__ == "__main__":

    # ----------------- read all data -----------------
    whole_data: List[Data] = data_repository.read_train_data(TRAIN_FILE_PATH)

    X = np.array([(d.date.hour,
                    d.date.day,
                    d.date.month,
                    d.date.year,
                    d.season,
                    d.holiday,
                    d.working_day,
                    d.weather,
                    d.temperature,
                    d.felt_temperature,
                    d.humidity,
                    d.wind_speed) for d in whole_data])

    casual = np.array([d.casual_cnt for d in whole_data])
    registered = np.array([d.registered_cnt for d in whole_data])
    total = np.array([d.total_cnt for d in whole_data])

    dates = np.array([d.date for d in whole_data])

    temperatures = np.array([d.temperature for d in whole_data])
    felt_temperatures = np.array([d.felt_temperature for d in whole_data])
    seasons = np.array([d.season for d in whole_data])

    humidities = np.array([d.humidity for d in whole_data])
    weathers = np.array([d.weather for d in whole_data])
    wind_speed = np.array([d.wind_speed for d in whole_data])

    # ----------------- evaluate counts correlation -----------------
    casual_registered_coeff = np.corrcoef(casual, registered)[1, 0]
    casual_total_coeff = np.corrcoef(casual, total)[1, 0]
    registered_total_coeff = np.corrcoef(registered, total)[1, 0]

    print("\nLe coefficient de correlation entre casual et registered est : {0:.3f}".format(casual_registered_coeff))
    print("Le coefficient de correlation entre casual et total est : {0:.3f}".format(casual_total_coeff))
    print("Le coefficient de correlation entre registered et total est : {0:.3f} \n".format(registered_total_coeff))

    # ----------------- evaluate intra features correlation -----------------

    n_features = X.shape[1]
    pairs = [(i, j) for i in range(n_features) for j in range(i + 1, n_features)]

    transposed_X = X.T

    intra_features_corr = np.ones((n_features, n_features))

    for pair in pairs:
        f1 = pair[0]
        f2 = pair[1]
        f1_vect = transposed_X[f1]
        f2_vect = transposed_X[f2]
        corr_coeff = np.corrcoef(f1_vect, f2_vect)[1, 0]
        corr_coeff = np.round(corr_coeff, 3)

        intra_features_corr[f1, f2] = corr_coeff
        intra_features_corr[f2, f1] = corr_coeff  # matrice symétrique

    headers = ("hour",
               "day",
               "month",
               "year",
               "season",
                "holiday",
                "working_day",
                "weather",
                "temperature",
                "felt_temperature",
                "humidity",
                "wind_speed")
    print("\nLa matrice des coefficients de correlation entre chaque paire de feature est : ")
    print(tabulate(intra_features_corr, headers, tablefmt='latex'))

    limit = 0.1
    print("\nLes paires de features ou on détecte une correlation intéressante (> à {}) sont: ".format(limit))
    print(tabulate(abs(intra_features_corr) > limit, headers))

    # ----------------- evaluate features vs counts correlation -----------------
    count_correlations = []
    for feat, head in zip(X.T, headers):
        casual_corr = np.corrcoef(feat, casual)[1, 0]
        registered_corr = np.corrcoef(feat, registered)[1, 0]
        count_correlations.append([head, np.round(casual_corr, 3), np.round(registered_corr, 3)])

    print("\nLes coefficients de corrélation entre chaque feature et le nombre de location casual et registered : ")
    print(tabulate(count_correlations, ("casual", "registered"), tablefmt='latex'))

    # ----------------- visualize -----------------
    fig, ax1 = pyplot.subplots()
    #pyplot.title("temperature, felt temperature and season vs time")

    color = 'tab:green'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Température', color=color)
    pyplot.scatter(dates, felt_temperatures, c='b', s=5, label="Température ressentie")
    pyplot.scatter(dates, temperatures, c='g', s=5, label="Véritable température")
    ax1.tick_params(axis='y', labelcolor=color)
    pyplot.legend()

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Saison', color=color)
    ax2.scatter(dates, seasons, c='r', s=5, label="saison")
    ax2.tick_params(axis='y', labelcolor=color)
    pyplot.yticks([1, 2, 3, 4])

    fig.tight_layout()
    pyplot.legend()
    pyplot.show()

    # fig, ax1 = pyplot.subplots()
    # pyplot.title("weather and humidity vs time")
    #
    # color = 'tab:red'
    # ax1.set_xlabel('date')
    # ax1.set_ylabel('weather', color=color)
    # ax1.scatter(dates, weathers, c='b', s=5, label="weather")
    # ax1.tick_params(axis='y', labelcolor=color)
    # pyplot.legend()
    #
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # color = 'tab:blue'
    # ax2.set_ylabel('humidity', color=color)  # we already handled the x-label with ax1
    # ax2.scatter(dates, humidities, c='g', s=5, label="humidity")
    # ax2.tick_params(axis='y', labelcolor=color)
    # pyplot.legend()
    #
    # ax3 = ax2.twinx()
    # color = 'tab:green'
    # ax2.set_ylabel('wind speed', color=color)  # we already handled the x-label with ax1
    # ax2.scatter(dates, wind_speed, c='y', s=5, label="wind speeds")
    # ax2.tick_params(axis='y', labelcolor=color)
    # pyplot.legend()
    #
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # pyplot.show()