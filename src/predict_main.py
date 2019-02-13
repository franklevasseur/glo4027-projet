import numpy as np
from typing import List
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

import data_repository
from data import Data
from constants import *


def format_data_for_prediction(d: Data):
    return (d.temperature,
            d.working_day,
            d.humidity)


if __name__ == "__main__":

    # ----------------- read all data -----------------
    whole_data: List[Data] = data_repository.read_train_data(TRAIN_FILE_PATH)

    dates = np.array([d.date for d in whole_data])

    casual_cnts = np.array([d.casual_cnt for d in whole_data])
    registered_cnts = np.array([d.registered_cnt for d in whole_data])

    # ----------------- Visualisation -----------------
    pyplot.figure()
    pyplot.title("locations par date")
    pyplot.scatter(dates, registered_cnts, c='g', s=5, label="registered counts")
    pyplot.scatter(dates, casual_cnts, c='r', s=5, label="casual counts")
    pyplot.legend()
    pyplot.show()

    # ----------------- Cleaning -----------------
    pass

    # ----------------- Training and prediction -----------------
    y = np.array([d.total_cnt for d in whole_data])
    X = np.array([format_data_for_prediction(d) for d in whole_data])

    data_size = len(whole_data)
    train_sample_size = data_size * 2 // 3

    train_indexes, test_indexes = train_test_split(np.arange(data_size), train_size=train_sample_size)
    X_train = X[train_indexes]
    X_test = X[test_indexes]
    y_train = y[train_indexes]
    y_test = y[test_indexes]

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_hat = model.predict(X_test)
    squared_deviations_sum = (np.square(y_hat - y_test)).sum()
    print("La somme des résidus carrés est : {0:,.2f}".format(squared_deviations_sum))

    test_dates = [d.date for i, d in enumerate(whole_data) if i in test_indexes]

    pyplot.figure()
    pyplot.title("Prédiction du compte total de location")
    pyplot.scatter(test_dates, y_test, c='g', s=5, label="actual")
    pyplot.scatter(test_dates, y_hat, c='r', s=5, label="predictions")
    pyplot.legend()
    pyplot.show()

    # ----------------- validation -----------------
    test_data: List[Data] = data_repository.read_test_data(TEST_FILE_PATH)
    X_submission = np.array([format_data_for_prediction(d) for d in test_data])
    y_submission = model.predict(X_submission)

    data_repository.write_submission_data(SUBMISSION_TEMPLATE_FILE_PATH, CURRENT_SUBMISSION, y_submission)



