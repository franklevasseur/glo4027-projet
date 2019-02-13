import numpy as np
from typing import List

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

import data_repository
import data_processing
from data import Data

from data_viewer import DataViewer

TRAIN_FILE_PATH = 'bike-sharing-demand/train.csv'
TEST_FILE_PATH = 'bike-sharing-demand/test.csv'
SUBMISSION_TEMPLATE_FILE_PATH = 'bike-sharing-demand/sampleSubmission.csv'
CURRENT_SUBMISSION = 'bike-sharing-demand/francoislevasseur_submission.csv'

VISUALISE = 0
VALIDATION = 0

if __name__ == "__main__":

    # ----------------- read all data -----------------
    whole_data: List[Data] = data_repository.read_train_data(TRAIN_FILE_PATH)

    dates = np.array([d.date for d in whole_data])

    months = np.array([d.date.month for d in whole_data])
    days = np.array([d.date.weekday() for d in whole_data])
    hours = np.array([d.date.hour for d in whole_data])
    seasons = np.array([d.season for d in whole_data])
    holidays = np.array([d.holiday for d in whole_data])
    working_days = np.array([d.working_day for d in whole_data])
    weathers = np.array([d.weather for d in whole_data])
    temperatures = np.array([d.temperature for d in whole_data])
    felt_temps = np.array([d.felt_temperature for d in whole_data])
    wind_speeds = np.array([d.wind_speed for d in whole_data])

    casual_cnts = np.array([d.casual_cnt for d in whole_data])
    registered_cnts = np.array([d.registered_cnt for d in whole_data])

    # ----------------- Visualisation -----------------
    viewer = DataViewer((registered_cnts, "locations de clients inscrits"), (casual_cnts, "locations casual"))
    viewer.plot_counts_vs_attributes(dates, "locations par date", save_as_pdf=False)

    if VISUALISE:
        pdf = True

        viewer.plot_counts_vs_attributes(months, "months", range(1, 13), save_as_pdf=pdf)
        viewer.plot_counts_vs_attributes(days, "days", range(0, 7), save_as_pdf=pdf)
        viewer.plot_counts_vs_attributes(hours, "hours", range(0, 24), save_as_pdf=pdf)
        viewer.plot_counts_vs_attributes(seasons, "seasons", range(1, 5), save_as_pdf=pdf)
        viewer.plot_counts_vs_attributes(holidays, "holidays", range(0, 2), save_as_pdf=pdf)
        viewer.plot_counts_vs_attributes(working_days, "working_days", range(0, 2), save_as_pdf=pdf)
        viewer.plot_counts_vs_attributes(weathers, "weathers", range(1, 5), save_as_pdf=pdf)
        viewer.plot_counts_vs_attributes(temperatures, "temperatures", save_as_pdf=pdf)
        viewer.plot_counts_vs_attributes(felt_temps, "felt_temps", save_as_pdf=pdf)
        viewer.plot_counts_vs_attributes(wind_speeds, "wind_speeds", save_as_pdf=pdf)

    # ----------------- Cleaning -----------------
    pass

    # ----------------- Training and prediction -----------------
    y = np.array([d.total_cnt for d in whole_data])
    X = np.array([data_processing.format_data_for_prediction(d) for d in whole_data])

    data_size = len(whole_data)
    train_sample_size = data_size * 2 // 3

    train_indexes, test_indexes = train_test_split(np.arange(data_size), train_size=train_sample_size)
    X_train = X[train_indexes]
    X_test = X[test_indexes]
    y_train = y[train_indexes]
    y_test = y[test_indexes]

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_hat = model.predict(X_test)
    squared_deviations_sum = (np.square(y_hat - y_test)).sum()
    print("La somme des résidus carrée est : {0:,.2f}".format(squared_deviations_sum))

    test_dates = [d.date for i, d in enumerate(whole_data) if i in test_indexes]

    viewer = DataViewer((y_test, "actual"), (y_hat, "predictions"))
    viewer.plot_counts_vs_attributes(test_dates, "Prédiction du compte total de location", save_as_pdf=False)

    # ----------------- validation -----------------
    if VALIDATION:
        test_data: List[Data] = data_repository.read_test_data(TEST_FILE_PATH)
        X_submission = np.array([data_processing.format_data_for_prediction(d) for d in test_data])
        y_submission = model.predict(X_submission)

        data_repository.write_submission_data(SUBMISSION_TEMPLATE_FILE_PATH, CURRENT_SUBMISSION, y_submission)



