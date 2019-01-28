import csv

from Data import Data


def read_file(file_path, format_row_func):
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        rows = []

        keys = next(csv_reader)
        for row in csv_reader:
            rows.append(format_row_func(row))

    return rows, keys


def read_train_data(file_path):
    format_row = lambda r: Data(True, r)
    data, _ = read_file(file_path, format_row)
    return data


def read_test_data(file_path):
    format_row = lambda r: Data(False, r)
    data, _ = read_file(file_path, format_row)
    return data



