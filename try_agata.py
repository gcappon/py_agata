import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from multiprocessing import freeze_support

from agata import Agata
from time_in_ranges.tests.test_time_in_hyperglycemia import test_time_in_hyperglycemia
def load_test_data(real=True, single_meal=True):
    if real:

        if single_meal:
            # Load real single meal data
            df = pd.read_csv(os.path.join('example', 'data', 'single-meal_example.csv'))
            df.t = pd.to_datetime(df['t'])
        else:
            # Load real multi meal data
            df = pd.read_csv(os.path.join('example', 'data', 'multi-meal_example.csv'))
            df.t = pd.to_datetime(df['t'])
    else:
        # Set fake data
        t = np.arange(datetime(2023, 1, 1, 6, 0, 0), datetime(2023, 1, 1, 12, 0, 0), timedelta(minutes=5)).astype(
            datetime)
        glucose = np.arange(360, 360 + t.size, 1)
        d = {'t': t, 'glucose': glucose}
        df = pd.DataFrame(data=d)

    return df


if __name__ == '__main__':

    freeze_support()

    # Get test data
    data = load_test_data(real=True, single_meal=True)

    glycemic_target = 'diabetes'

    agata = Agata(data=data, glycemic_target=glycemic_target)

    #Analyze data
    res = agata.analyze_glucose_profile()
    print(res)

    test_time_in_hyperglycemia()
