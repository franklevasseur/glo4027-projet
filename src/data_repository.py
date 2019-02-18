import csv

from src.data import Data


def read_file(file_path, format_row_func):
    with open(file_path, mode='r') as csv_file:
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


def write_submission_data(template_file_path, current_submission, predictions):
    fn = lambda r: r
    raw_submission_file_content, keys = read_file(template_file_path, fn)

    for line, prediction in zip(raw_submission_file_content, predictions):
        line[1] = prediction

    with open(current_submission, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')

        csv_writer.writerow(keys)
        for line in raw_submission_file_content:
            csv_writer.writerow(line)


