import data_reader

DAY_FILE_PATH = 'Bike-Sharing-Dataset/day.csv'
HOUR_FILE_PATH = 'Bike-Sharing-Dataset/hour.csv'

if __name__ == "__main__":
    print('Program start')
    day_keys, day_data = data_reader.read_file(DAY_FILE_PATH)

    print("keys are {} \n data is...".format(day_keys))
    for row in day_data:
        print(row)


