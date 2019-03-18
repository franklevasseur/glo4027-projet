import numpy as np
from typing import List
from statistics import stdev

from tabulate import tabulate

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import data_repository
from data import Data
from constants import *

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


def train_and_predict(model_casual, model_registered, Xcasual, Ycasual,
                      Xregistered, Yregistered, train_indexes, test_indexes):
    Xcasual_train = Xcasual[train_indexes]
    Xcasual_test = Xcasual[test_indexes]
    Xregistered_train = Xregistered[train_indexes]
    Xregistered_test = Xregistered[test_indexes]

    Ycasual_train = Ycasual[train_indexes]
    Yregistered_train = Yregistered[train_indexes]
    Y_test = Ycasual[test_indexes] + Yregistered[test_indexes]

    model_casual.fit(Xcasual_train, Ycasual_train)
    model_registered.fit(Xregistered_train, Yregistered_train)

    Y_hat = model_casual.predict(Xcasual_test) + model_registered.predict(Xregistered_test)
    Y_hat[Y_hat < 0] = 0

    squared_deviations_sum = (np.square(Y_hat - Y_test)).sum()
    kaggle_score = get_kaggle_score(Y_test, Y_hat)

    return kaggle_score, squared_deviations_sum


def build_forest(n_estimators, max_depth, max_features):
    return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)


if __name__ == "__main__":

    # ----------------- read all data -----------------
    train_data: List[Data] = data_repository.read_train_data(TRAIN_FILE_PATH)

    dates = np.array([d.date for d in train_data])

    casual_cnts = np.array([d.casual_cnt for d in train_data])
    registered_cnts = np.array([d.registered_cnt for d in train_data])

    # ----------------- Training and prediction -----------------
    Ycasual = np.array([d.casual_cnt for d in train_data])
    Yregistered = np.array([d.registered_cnt for d in train_data])
    Xcasual = np.array([format_data_for_casual_prediction(d) for d in train_data])
    Xregistered = np.array([format_data_for_registered_prediction(d) for d in train_data])

    data_size = len(train_data)
    train_sample_size = data_size * 2 // 3

    n_estimators = [10, 50, 100]
    max_features_per_node = [0.2, 0.4, 0.6, 0.8, None]
    max_depths = [3, 6, 9, 12]
    clf_results = [["" for i in range(len(max_features_per_node))] for j in range(len(max_depths))]
    n = 10
    for k, n_estimator in enumerate(n_estimators):
        for j, max_feat in enumerate(max_features_per_node):
            for i, max_depth in enumerate(max_depths):

                kaggle_scores = []
                for test in range(n):
                    train_indexes, test_indexes = train_test_split(np.arange(data_size), train_size=train_sample_size)

                    kaggle_score, _ = train_and_predict(build_forest(n_estimator, max_depth, max_feat),
                                                        build_forest(n_estimator, max_depth, max_feat),
                                                             Xcasual,
                                                             Ycasual,
                                                             Xregistered,
                                                             Yregistered,
                                                             train_indexes,
                                                             test_indexes)
                    kaggle_scores.append(kaggle_score)

                mean_kaggle_score = sum(kaggle_scores) / n
                std_kaggle_score = stdev(kaggle_scores)
                mean_kaggle_score = round(mean_kaggle_score, 3)
                std_kaggle_score = round(std_kaggle_score, 3)

                formated_score = "{:.3f} , {:.3f}".format(mean_kaggle_score, std_kaggle_score)

                clf_results[i][j] = formated_score

        for (r, row), mx in zip(enumerate(clf_results), max_depths):
            formated_row = [str(mx)]
            formated_row.extend(row)
            clf_results[r] = formated_row

        print("nombre d'arbres = {}".format(n_estimator))
        print(tabulate(clf_results, max_features_per_node, tablefmt='latex'))
        clf_results = [["" for i in range(len(max_features_per_node))] for j in range(len(max_depths))]
