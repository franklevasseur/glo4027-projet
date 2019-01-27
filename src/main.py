import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC

import data_reader

DAY_FILE_PATH = 'Bike-Sharing-Dataset/day.csv'
HOUR_FILE_PATH = 'Bike-Sharing-Dataset/hour.csv'

if __name__ == "__main__":

    daily_data = data_reader.read_day_data(DAY_FILE_PATH)
    # hourly_data = data_reader.read_day_data(HOUR_FILE_PATH)

    Y = np.array([d.casual_cnt for d in daily_data])
    X = np.array([(d.year,
                 d.month,
                 d.holiday,
                 d.weekday,
                 d.weather,
                 d.temperature,
                 d.humidity, d.wind_speed) for d in daily_data])

    model = LinearSVC()
    model.fit(X, Y)

    N = 3
    selector = RFE(model, n_features_to_select=N, step=1)
    selector.fit(X, Y)

    kept_features = [i for (i, r) in enumerate(selector.ranking_) if r == 1]

    print("les features les plus significatifs selon la sélection arrière séquentielle sont {}".format(kept_features))

    x_temp = np.array([d.temperature for d in daily_data])

    plt.scatter(x_temp, Y)
    plt.show()
