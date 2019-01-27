import csv


def read_file(file_path):
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        keys = []
        rows = []
        for i, row in enumerate(csv_reader):
            if i == 0:
                keys.extend(row)
            else:
                rows.append(row)

    return keys, rows


def read_data(file_path):
    pass
