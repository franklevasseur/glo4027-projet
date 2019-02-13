from data import Data


def format_data_for_prediction(d: Data):
    return (d.temperature,
                   d.working_day,
                   d.humidity)