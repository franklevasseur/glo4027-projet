import numpy as np
from typing import List
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

import data_repository
from data import Data
from constants import *

VISUALIZE = 0


def format_data_for_casual_prediction(d: Data):
    return (d.date.hour,
            d.date.month,
            d.season,
            d.holiday,
            d.working_day,
            d.weather,
            d.temperature,
            d.felt_temperature,
            d.humidity,
            d.wind_speed)


def format_data_for_registered_prediction(d: Data):
    return (d.date.hour,
            d.date.month,
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


if __name__ == "__main__":

    # ----------------- read all data -----------------
    train_data: List[Data] = data_repository.read_train_data(TRAIN_FILE_PATH)

    dates = np.array([d.date for d in train_data])

    casual_cnts = np.array([d.casual_cnt for d in train_data])
    registered_cnts = np.array([d.registered_cnt for d in train_data])

    # ----------------- Cleaning -----------------
    pass

    # ----------------- Training and prediction -----------------
    Ycasual = np.array([d.casual_cnt for d in train_data])
    Yregistered = np.array([d.registered_cnt for d in train_data])
    Xcasual = np.array([format_data_for_casual_prediction(d) for d in train_data])
    Xregistered = np.array([format_data_for_registered_prediction(d) for d in train_data])

    data_size = len(train_data)
    train_sample_size = data_size * 2 // 3

    train_indexes, test_indexes = train_test_split(np.arange(data_size), train_size=train_sample_size)
    Xcasual_train = Xcasual[train_indexes]
    Xcasual_test = Xcasual[test_indexes]
    Xregistered_train = Xregistered[train_indexes]
    Xregistered_test = Xregistered[test_indexes]

    Ycasual_train = Ycasual[train_indexes]
    Yregistered_train = Yregistered[train_indexes]
    Y_test = Ycasual[test_indexes] + Yregistered[test_indexes]

    model1 = RandomForestRegressor()
    model1.fit(Xcasual_train, Ycasual_train)

    model2 = RandomForestRegressor()
    model2.fit(Xregistered_train, Yregistered_train)

    Y_hat = model1.predict(Xcasual_test) + model2.predict(Xregistered_test)

    squared_deviations_sum = (np.square(Y_hat - Y_test)).sum()
    print("La somme des résidus carrés est : {0:,.2f}".format(squared_deviations_sum))

    print("Le score Kaggle est : {0:.3f}".format(get_kaggle_score(Y_test, Y_hat)))

    test_dates = [d.date for i, d in enumerate(train_data) if i in test_indexes]

    if VISUALIZE:
        pyplot.figure()
        pyplot.title("Prédiction du compte total de location")
        pyplot.scatter(test_dates, Y_test, c='g', s=5, label="actual")
        pyplot.scatter(test_dates, Y_hat, c='r', s=5, label="predictions")
        pyplot.legend()
        pyplot.show()

    # ----------------- validation -----------------
    # model1.fit(Xcasual, Ycasual)  # on a gaspillé le 1/3 du dataset tentot alors ici on le reprend au complet
    # model2.fit(Xregistered, Yregistered)

    X_something = np.array([(d.date.hour,
            d.season,
            d.holiday,
            d.working_day,
            d.weather,
            d.temperature,
            d.felt_temperature,
            d.humidity,
            d.wind_speed) for d in train_data])
    Y_something = np.array([d.total_cnt for d in train_data])

    model = RandomForestRegressor()
    model.fit(X_something, Y_something)

    test_data: List[Data] = data_repository.read_test_data(TEST_FILE_PATH)
    # X1_submission = np.array([format_data_for_casual_prediction(d) for d in test_data])
    # X2_submission = np.array([format_data_for_registered_prediction(d) for d in test_data])
    X_submission = np.array([(d.date.hour,
            d.season,
            d.holiday,
            d.working_day,
            d.weather,
            d.temperature,
            d.felt_temperature,
            d.humidity,
            d.wind_speed) for d in test_data])

    y_submission = model.predict(X_submission)

    data_repository.write_submission_data(SUBMISSION_TEMPLATE_FILE_PATH, CURRENT_SUBMISSION, y_submission)



