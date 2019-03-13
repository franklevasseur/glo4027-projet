import numpy as np
from typing import List

from tabulate import tabulate

from sklearn.ensemble import RandomForestRegressor

import data_repository
from data import Data
from constants import *


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


def get_feature_ranking(casual_forest, registered_forest, X_dataset):
    casual_importances = casual_forest.feature_importances_
    casual_indices = np.argsort(casual_importances)[::-1]

    registered_importances = registered_forest.feature_importances_
    registered_indices = np.argsort(casual_importances)[::-1]

    results = []
    for f in range(X_dataset.shape[1]):
        row = [f + 1]
        row.append("{} ({:3f})".format(casual_indices[f], casual_importances[casual_indices[f]]))
        row.append("{} ({:3f})".format(registered_indices[f], registered_importances[registered_indices[f]]))
        results.append(row)

    print(tabulate(results, ("ordre", "occasionnel", "inscrit"), tablefmt='latex'))

    return results


def create_model():
    return RandomForestRegressor(bootstrap=False, n_estimators=10, max_depth=10, max_features=0.8)


def train_and_predict(Xcasual, Ycasual, Xregistered, Yregistered):
    model_casual = create_model()
    model_casual.fit(Xcasual, Ycasual)

    model_registered = create_model()
    model_registered.fit(Xregistered, Yregistered)

    get_feature_ranking(model_casual, model_registered, Xcasual)


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

    mean_kaggle_score, mean_squared_residal_sum = 0, 0
    n = 10
    for i in range(n):
        train_and_predict(Xcasual, Ycasual, Xregistered, Yregistered)
