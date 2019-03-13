import numpy as np
from typing import List
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

import data_repository
from data import Data
from constants import *

KAGGLE_VALIDATION = 0

def format_data_for_casual_prediction(d: Data):
    return (d.date.hour,
            # d.date.month,
            d.season,
            d.holiday,
            d.working_day,
            d.weather,
            d.temperature,
            # d.felt_temperature,
            d.humidity,
            d.wind_speed)


def format_data_for_registered_prediction(d: Data):
    return (d.date.hour,
            # d.date.month,
            d.season,
            d.holiday,
            d.working_day,
            d.weather,
            d.temperature,
            # d.felt_temperature,
            d.humidity,
            d.wind_speed)


def get_kaggle_score(actual, prediction):
    n = actual.size
    score = np.log(prediction + 1) - np.log(actual + 1)
    score = np.square(score)
    score = (1 / n) * score.sum()
    score = np.sqrt(score)

    return score


def get_feature_ranking(forest: RandomForestRegressor, X_dataset, dataset_name=""):
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking {}:".format(dataset_name))

    for f in range(X_dataset.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


def create_model():
    return RandomForestRegressor(bootstrap=False, n_estimators=10, max_depth=10, max_features=0.8)


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

    # get_feature_ranking(model_casual, Xcasual, "causal")
    # get_feature_ranking(model_registered, Xregistered, "registered")

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
    train_data = np.array([d for d in train_data if d.weather != 4])

    # ----------------- Training and prediction -----------------
    Ycasual = np.array([d.casual_cnt for d in train_data])
    Yregistered = np.array([d.registered_cnt for d in train_data])
    Xcasual = np.array([format_data_for_casual_prediction(d) for d in train_data])
    Xregistered = np.array([format_data_for_registered_prediction(d) for d in train_data])

    data_size = len(train_data)
    train_sample_size = data_size * 2 // 3

    mean_kaggle_score, mean_squared_residal_sum = 0, 0
    n = 10
    for i in range(n):
        train_indexes, test_indexes = train_test_split(np.arange(data_size), train_size=train_sample_size)

        kaggle_score, squared_deviations_sum = train_and_predict(Xcasual,
                                                                 Ycasual,
                                                                 Xregistered,
                                                                 Yregistered,
                                                                 train_indexes,
                                                                 test_indexes)
        mean_kaggle_score += kaggle_score / n
        mean_squared_residal_sum += squared_deviations_sum / n

    print("La somme moyenne des résidus carrés sur {0} essais est : {1:,.2f}".format(n, mean_squared_residal_sum))
    print("Le score moyen Kaggle sur {0} essais est : {1:.3f}".format(n, mean_kaggle_score))

    if KAGGLE_VALIDATION:
        # ----------------- validation -----------------
        kaggle_model_casual = create_model()
        kaggle_model_registered = create_model()

        # on a gaspillé le 1/3 du dataset tentot alors ici on le reprend au complet
        kaggle_model_casual.fit(Xcasual, Ycasual)
        kaggle_model_registered.fit(Xregistered, Yregistered)

        test_data: List[Data] = data_repository.read_test_data(TEST_FILE_PATH)
        X1_submission = np.array([format_data_for_casual_prediction(d) for d in test_data])
        X2_submission = np.array([format_data_for_registered_prediction(d) for d in test_data])

        y_submission = kaggle_model_casual.predict(X1_submission) + kaggle_model_registered.predict(X2_submission)

        data_repository.write_submission_data(SUBMISSION_TEMPLATE_FILE_PATH, CURRENT_SUBMISSION, y_submission)



