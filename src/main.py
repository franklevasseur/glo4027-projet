import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVR

import data_reader

TRAIN_FILE_PATH = 'bike-sharing-demand/train.csv'
TEST_FILE_PATH = 'bike-sharing-demand/test.csv'

if __name__ == "__main__":

    train_data = data_reader.read_train_data(TRAIN_FILE_PATH)

    Y = np.array([d.total_cnt for d in train_data])
    X = np.array([(d.date.year,
                 d.date.month,
                 d.holiday,
                 d.working_day,
                 d.weather,
                 d.temperature,
                 d.humidity, d.wind_speed) for d in train_data])

    model = LinearSVR()
    model.fit(X, Y)

    N = 3
    selector = RFE(model, n_features_to_select=N, step=1)
    selector.fit(X, Y)

    kept_features = [i for (i, r) in enumerate(selector.ranking_) if r == 1]

    print("les features les plus significatives selon la sélection arrière séquentielle sont {}".format(kept_features))

    x_temp = np.array([d.temperature for d in train_data])

    plt.scatter(x_temp, Y)
    plt.show()
