import numpy as np
from matplotlib import pyplot
from typing import List

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

import data_reader
from Data import Data

TRAIN_FILE_PATH = 'bike-sharing-demand/train.csv'
TEST_FILE_PATH = 'bike-sharing-demand/test.csv'

VISUALISE = 0

if __name__ == "__main__":

    # read all data
    whole_data: List[Data] = data_reader.read_train_data(TRAIN_FILE_PATH)

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

    # Visualisation
    if VISUALISE:
        def plot_attribute(p_vect, p_xlabel, p_xtick=None):
            pyplot.figure()
            pyplot.xlabel(p_xlabel)
            pyplot.scatter(p_vect, registered_cnts, c='g', s=5, label="registered counts")
            pyplot.scatter(p_vect, casual_cnts, c='r', s=50, marker="*", label="casual counts")

            if p_xtick:
                pyplot.xticks(p_xtick)

            pyplot.legend()
            pyplot.savefig("./figs/{}.pdf".format(p_xlabel))

        plot_attribute(months, "months", range(1, 13))
        plot_attribute(days, "days", range(0, 7))
        plot_attribute(hours, "hours", range(0, 24))
        plot_attribute(seasons, "seasons", range(1, 5))
        plot_attribute(holidays, "holidays", range(0, 2))
        plot_attribute(working_days, "working_days", range(0, 2))
        plot_attribute(weathers, "weathers", range(1, 5))
        plot_attribute(temperatures, "temperatures")
        plot_attribute(felt_temps, "felt_temps")
        plot_attribute(wind_speeds, "wind_speeds")

    # Cleaning


    # Training and prediction
    y = np.array([d.total_cnt for d in whole_data])
    X = np.array([(d.temperature,
                   d.working_day,
                   d.humidity) for d in whole_data])

    data_size = len(whole_data)
    train_sample_size = data_size * 2 // 3

    train_indexes, test_indexes = train_test_split(np.arange(data_size), train_size=train_sample_size)
    X_train = X[train_indexes]
    X_test = X[test_indexes]
    y_train = y[train_indexes]
    y_test = y[test_indexes]

    model = LinearSVR()
    model.fit(X_train, y_train)

    y_hat = model.predict(X_test)
    squared_deviations_sum = (np.square(y_hat - y_test)).sum()

    test_dates = [d.date for i, d in enumerate(whole_data) if i in test_indexes]

    pyplot.figure()
    pyplot.xlabel('date')
    pyplot.ylabel('nombre de location')
    pyplot.scatter(test_dates, y_hat, c='g', s=5, label="prediction")
    pyplot.scatter(test_dates, y_test, c='r', s=5, marker="*", label="actual")
    pyplot.legend()
    pyplot.show()

    print("La somme des résidus carrée est : {0:.2f}".format(squared_deviations_sum))
