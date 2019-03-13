import numpy as np
from typing import List
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

import data_repository
from data import Data
from constants import *

KAGGLE_VALIDATION = 1


def format_data_for_prediction(d: Data):
    return (d.date.hour,
            d.date.month,
            d.date.year,
            d.season,
            d.holiday,
            d.working_day,
            d.weather,
            d.temperature,
            d.felt_temperature,
            d.humidity,
            d.wind_speed)


def get_kaggle_score(actual, prediction):
    n = actual.size
    score = np.log(prediction + 1) - np.log(actual + 1)
    score = np.square(score)
    score = (1 / n) * score.sum()
    score = np.sqrt(score)

    return score


def create_model():
    return RandomForestRegressor(n_estimators=100)


if __name__ == "__main__":

    # ----------------- read all data -----------------
    train_data: List[Data] = data_repository.read_train_data(TRAIN_FILE_PATH)

    dates = np.array([d.date for d in train_data])

    casual_cnts = np.array([d.casual_cnt for d in train_data])
    registered_cnts = np.array([d.registered_cnt for d in train_data])

    # ----------------- Cleaning -----------------
    pass

    # ----------------- Training and prediction -----------------
    X = np.array([format_data_for_prediction(d) for d in train_data])
    Y = np.array([d.total_cnt for d in train_data])

    data_size = len(train_data)
    train_sample_size = data_size * 2 // 3

    train_indexes, test_indexes = train_test_split(np.arange(data_size), train_size=train_sample_size)
    X_train = X[train_indexes]
    X_test = X[test_indexes]

    Y_train = Y[train_indexes]
    Y_test = Y[test_indexes]

    model = create_model()
    model.fit(X_train, Y_train)
    Y_hat = model.predict(X_test)

    squared_deviations_sum = (np.square(Y_hat - Y_test)).sum()
    print("La somme des résidus carrés est : {0:,.2f}".format(squared_deviations_sum))
    print("Le score Kaggle est : {0:.3f}".format(get_kaggle_score(Y_test, Y_hat)))

    if KAGGLE_VALIDATION:
        # ----------------- validation -----------------
        kaggle_model = create_model()

        # on a gaspillé le 1/3 du dataset tentot alors ici on le reprend au complet
        kaggle_model.fit(X, Y)

        test_data: List[Data] = data_repository.read_test_data(TEST_FILE_PATH)
        X_submission = np.array([format_data_for_prediction(d) for d in test_data])

        y_submission = kaggle_model.predict(X_submission)

        data_repository.write_submission_data(SUBMISSION_TEMPLATE_FILE_PATH, CURRENT_SUBMISSION, y_submission)

