import numpy as np
from typing import List
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

import data_repository
from data import Data
from constants import *


def format_data_for_casual_prediction(d: Data):
    return (d.date.hour,
            d.temperature,
            d.working_day,
            d.humidity)


def format_data_for_registered_prediction(d: Data):
    return (d.date.hour,
            d.temperature,
            d.humidity)


def get_kaggle_score(actual, prediction):
    n = actual.size
    score = np.log(prediction + 1) - np.log(actual + 1)
    score = np.square(score)
    score = (1 / n) * score.sum()
    score = np.sqrt(score)

    return score


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
    y1 = np.array([d.casual_cnt for d in whole_data])
    y2 = np.array([d.registered_cnt for d in whole_data])
    X1 = np.array([format_data_for_casual_prediction(d) for d in whole_data])
    X2 = np.array([format_data_for_registered_prediction(d) for d in whole_data])

    data_size = len(whole_data)
    train_sample_size = data_size * 2 // 3

    train_indexes, test_indexes = train_test_split(np.arange(data_size), train_size=train_sample_size)
    X1_train = X1[train_indexes]
    X1_test = X1[test_indexes]
    X2_train = X2[train_indexes]
    X2_test = X2[test_indexes]

    y1_train = y1[train_indexes]
    y2_train = y2[train_indexes]
    y_test = y1[test_indexes] + y2[test_indexes]

    model1 = RandomForestRegressor()
    model1.fit(X1_train, y1_train)

    model2 = RandomForestRegressor()
    model2.fit(X2_train, y2_train)

    y_hat = model1.predict(X1_test) + model2.predict(X2_test)

    squared_deviations_sum = (np.square(y_hat - y_test)).sum()
    print("La somme des résidus carrés est : {0:,.2f}".format(squared_deviations_sum))

    print("Le score Kaggle est : {0:.3f}".format(get_kaggle_score(y_test, y_hat)))

    test_dates = [d.date for i, d in enumerate(whole_data) if i in test_indexes]

    pyplot.figure()
    pyplot.title("Prédiction du compte total de location")
    pyplot.scatter(test_dates, y_test, c='g', s=5, label="actual")
    pyplot.scatter(test_dates, y_hat, c='r', s=5, label="predictions")
    pyplot.legend()
    pyplot.show()

    # ----------------- validation -----------------
    model1.fit(X1, y1)  # on a gaspillé le 1/3 du dataset tentot pour l'entrainement alors ici on le reprend au complet
    model2.fit(X2, y2)

    test_data: List[Data] = data_repository.read_test_data(TEST_FILE_PATH)
    X1_submission = np.array([format_data_for_casual_prediction(d) for d in test_data])
    X2_submission = np.array([format_data_for_registered_prediction(d) for d in test_data])

    y_submission = model1.predict(X1_submission) + model2.predict(X2_submission)

    data_repository.write_submission_data(SUBMISSION_TEMPLATE_FILE_PATH, CURRENT_SUBMISSION, y_submission)



