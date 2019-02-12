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

if __name__ == "__main__":

    whole_data: List[Data] = data_reader.read_train_data(TRAIN_FILE_PATH)
    y = np.array([d.total_cnt for d in whole_data])
    X = np.array([(d.date.month,
            d.date.weekday(),
            d.date.hour,
            d.season,
            d.holiday,
            d.working_day,
            d.weather,
            d.temperature,
            d.felt_temperature,
            d.humidity,
            d.wind_speed) for d in whole_data])

    names = ["date.month", "date.day", "date.hour", "season", "holiday", "working_day", "weather",
             "temperature", "felt_temperature", "wind_speed"]

    # visualisation
    fig, subfigs = pyplot.subplots(3, 4)
    for feat, subfig, name in zip(X.T, subfigs.reshape(-1), names):
        subfig.set_xlabel(name)
        subfig.scatter(feat, y, s=1)

    pyplot.subplots_adjust(hspace=0.6)
    pyplot.show()

    # Training and prediction
    y = np.array([d.total_cnt for d in whole_data])
    X = np.array([(d.date.month,
                   d.date.weekday(),
                   d.date.hour,
                   d.season,
                   d.holiday,
                   d.working_day,
                   d.weather,
                   d.temperature,
                   d.felt_temperature,
                   d.humidity,
                   d.wind_speed) for d in whole_data])

    train_sample_size = len(whole_data) * 2 // 3

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_sample_size)

    model = LinearSVR()
    model.fit(X_train, y_train)

    y_hat = model.predict(X_test)
    squared_deviations_sum = (np.square(y_hat - y_test)).sum()

    print("La somme des résidus carrée est : {}".format(squared_deviations_sum))
