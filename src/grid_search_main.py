import numpy as np
from typing import List
from statistics import stdev
from matplotlib import pyplot

from tabulate import tabulate

from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from random import randint

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


def build_knn(n_neighbors, distance_power):
    return KNeighborsRegressor(n_neighbors=n_neighbors, metric=distance_power)


def get_column(two_dim_list, column, lenght):
    return [two_dim_list[i][column] for i in range(lenght)]


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

    Xcasual = preprocessing.scale(Xcasual)
    Xregistered = preprocessing.scale(Xregistered)

    n_neighbors_counts = [1, 2, 3, 4]
    n_neighbors_counts.extend([n for n in range(5, 105, 5)])
    distances = ["euclidean", "manhattan", "chebyshev"]
    clf_results = [[None for i in range(len(distances))] for j in range(len(n_neighbors_counts))]
    clf_results_formatted = [["" for i in range(len(distances))] for j in range(len(n_neighbors_counts))]
    n = 10
    for i, n_neighbors in enumerate(n_neighbors_counts):
        for j, distance_metric in enumerate(distances):
            print("n = {}, p = {}".format(n_neighbors, distance_metric))

            kaggle_scores = []

            for test in range(n):

                test_days_begin = randint(0, 15)
                test_days = range(test_days_begin, test_days_begin + 5)  # on veut prédire 5 jours consécutifs
                train_indexes = [i for i, d in enumerate(train_data) if d.date.day not in test_days]
                test_indexes = [i for i, d in enumerate(train_data) if d.date.day in test_days]

                kaggle_score, _ = train_and_predict(build_knn(n_neighbors, distance_metric),
                                                    build_knn(n_neighbors, distance_metric),
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

            clf_results[i][j] = mean_kaggle_score
            clf_results_formatted[i][j] = formated_score

    pyplot.figure()
    pyplot.ylabel("Score Kaggle moyen sur {} essais".format(n))
    pyplot.xlabel("Nombre de voisins")
    for i, p in enumerate(distances):
        pyplot.plot(n_neighbors_counts, get_column(clf_results, i, len(n_neighbors_counts)), label="p = {}".format(p))

    pyplot.legend()
    pyplot.show()

    for (r, row), n in zip(enumerate(clf_results_formatted), n_neighbors_counts):
        formatted_row = ["n = {}".format(n)]
        formatted_row.extend(row)
        clf_results_formatted[r] = formatted_row

    print(tabulate(clf_results_formatted, distances, tablefmt='latex'))