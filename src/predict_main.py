import numpy as np
from typing import List
from matplotlib import pyplot
from statistics import stdev

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

import data_repository
from data import Data
from constants import *

N = 10
KAGGLE_VALIDATION = 0

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def format_data_for_casual_prediction(d: Data):
    return (d.date.hour,
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
            d.wind_speed)


def format_data_for_registered_prediction(d: Data):
    return (d.date.hour,
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
            d.wind_speed)


def get_kaggle_score(actual, prediction):
    n = actual.size
    score = np.log(prediction + 1) - np.log(actual + 1)
    score = np.square(score)
    score = (1 / n) * score.sum()
    score = np.sqrt(score)

    return score


def create_model():
    return RandomForestRegressor(n_estimators=100, max_depth=12, max_features=None)


def train_and_predict(Xcasual, Ycasual, Xregistered, Yregistered, train_indexes, test_indexes):
    Xcasual_train = Xcasual[train_indexes]
    Xcasual_test = Xcasual[test_indexes]
    Xregistered_train = Xregistered[train_indexes]
    Xregistered_test = Xregistered[test_indexes]

    Ycasual_train = Ycasual[train_indexes]
    Yregistered_train = Yregistered[train_indexes]
    Y_test = Ycasual[test_indexes] + Yregistered[test_indexes]

    model_casual = create_model()
    model_casual.fit(Xcasual_train, Ycasual_train)

    model_registered = create_model()
    model_registered.fit(Xregistered_train, Yregistered_train)

    Y_hat = model_casual.predict(Xcasual_test) + model_registered.predict(Xregistered_test)

    squared_deviations_sum = (np.square(Y_hat - Y_test)).sum()
    kaggle_score = get_kaggle_score(Y_test, Y_hat)

    return kaggle_score, squared_deviations_sum


if __name__ == "__main__":

    # ----------------- read all data -----------------
    train_data: List[Data] = data_repository.read_train_data(TRAIN_FILE_PATH)

    dates = np.array([d.date for d in train_data])

    casual_cnts = np.array([d.casual_cnt for d in train_data])
    registered_cnts = np.array([d.registered_cnt for d in train_data])

    # ----------------- Cleaning -----------------
    # train_data = np.array([d for d in train_data if d.weather != 4])
    # train_data = np.array([d for d in train_data if d.humidity != 0])

    # ----------------- Training and prediction -----------------
    Ycasual = np.array([d.casual_cnt for d in train_data])
    Yregistered = np.array([d.registered_cnt for d in train_data])
    Xcasual = np.array([format_data_for_casual_prediction(d) for d in train_data])
    Xregistered = np.array([format_data_for_registered_prediction(d) for d in train_data])

    data_size = len(train_data)
    train_sample_size = data_size * 2 // 3

    mean_kaggle_score, mean_squared_residal_sum = 0, 0
    scores = []
    for i in range(N):
        train_indexes, test_indexes = train_test_split(np.arange(data_size), train_size=train_sample_size)

        kaggle_score, squared_deviations_sum = train_and_predict(Xcasual,
                                                                 Ycasual,
                                                                 Xregistered,
                                                                 Yregistered,
                                                                 train_indexes,
                                                                 test_indexes)
        scores.append(kaggle_score)
        mean_squared_residal_sum += squared_deviations_sum / N

    mean_kaggle_score = sum(scores) / N
    stdev_kaggle_score = stdev(scores)

    print("La somme moyenne des résidus carrés sur {0} essais est : {1:,.2f}".format(N, mean_squared_residal_sum))
    print("Le score moyen Kaggle sur {0} essais est : {1:.3f} ± {2}".format(N, mean_kaggle_score, stdev_kaggle_score))

    if KAGGLE_VALIDATION:
        # ----------------- validation -----------------
        kaggle_model_casual = create_model()
        kaggle_model_registered = create_model()

        # on a gaspillé le 1/3 du dataset tentot alors ici on le reprend au complet
        kaggle_model_casual.fit(Xcasual, Ycasual)
        kaggle_model_registered.fit(Xregistered, Yregistered)

        test_data: List[Data] = data_repository.read_test_data(TEST_FILE_PATH)
        Xcasual_submission = np.array([format_data_for_casual_prediction(d) for d in test_data])
        Xregistered_submission = np.array([format_data_for_registered_prediction(d) for d in test_data])

        y_submission = kaggle_model_casual.predict(Xcasual_submission) \
                       + kaggle_model_registered.predict(Xregistered_submission)

        data_repository.write_submission_data(SUBMISSION_TEMPLATE_FILE_PATH, CURRENT_SUBMISSION, y_submission)



