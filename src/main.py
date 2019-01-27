import data_reader

DAY_FILE_PATH = 'Bike-Sharing-Dataset/day.csv'
HOUR_FILE_PATH = 'Bike-Sharing-Dataset/hour.csv'

if __name__ == "__main__":
    
    daily_data = data_reader.read_day_data(DAY_FILE_PATH)
    hourly_data = data_reader.read_day_data(HOUR_FILE_PATH)


