TRAIN_FILE_PATH = '../bike-sharing-demand/train.csv'
TEST_FILE_PATH = '../bike-sharing-demand/test.csv'
SUBMISSION_TEMPLATE_FILE_PATH = '../bike-sharing-demand/sampleSubmission.csv'
CURRENT_SUBMISSION = '../bike-sharing-demand/francoislevasseur_submission.csv'

# csv file indexes
DATE = 0
SEASON = DATE + 1
HOLIDAY = SEASON + 1
WORKING_DAY = HOLIDAY + 1
WEATHER = WORKING_DAY + 1
TEMP = WEATHER + 1
FELT_TEMP = TEMP + 1
HUMIDITY = FELT_TEMP + 1
WIND_SPEED = HUMIDITY + 1
CASUAL_CNT = WIND_SPEED + 1
REGISTERED_CNT = CASUAL_CNT + 1
TOTAL_CNT = REGISTERED_CNT + 1