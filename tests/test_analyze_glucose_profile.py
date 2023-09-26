import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from agata import Agata


def test_agata_init():
    # Set test data
    t = np.arange(datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 1, 0, 55, 0), timedelta(minutes=5)).astype(
        datetime)
    glucose = np.zeros(shape=(t.shape[0],))
    glucose[0] = 40
    glucose[1:3] = [60, 60]
    glucose[3] = 80
    glucose[4:6] = [120, 150]
    glucose[6:8] = [200, 200]
    glucose[8:10] = [260, 260]
    glucose[10] = np.nan
    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    #Tests

    #Diabetes
    agata = Agata(data=data, glycemic_target='diabetes')

    results = agata.analyze_glucose_profile()

    #Time fields
    assert type(results) is dict
    assert type(results['time']) is dict
    assert results['time']['time_in_target'] == 30
    assert results['time']['time_in_tight_target'] == 20
    assert results['time']['time_in_hypoglycemia'] == 30
    assert results['time']['time_in_l1_hypoglycemia'] == 20
    assert results['time']['time_in_l2_hypoglycemia'] == 10
    assert results['time']['time_in_hyperglycemia'] == 40
    assert results['time']['time_in_l1_hyperglycemia'] == 20
    assert results['time']['time_in_l2_hyperglycemia'] == 20

    #Pregnancy
    agata = Agata(data=data, glycemic_target='pregnancy')

    results = agata.analyze_glucose_profile()

    # Time fields
    assert type(results) is dict
    assert type(results['time']) is dict
    assert results['time']['time_in_target'] == 20
    assert results['time']['time_in_tight_target'] == 20
    assert results['time']['time_in_hypoglycemia'] == 30
    assert results['time']['time_in_l1_hypoglycemia'] == 20
    assert results['time']['time_in_l2_hypoglycemia'] == 10
    assert results['time']['time_in_hyperglycemia'] == 50
    assert results['time']['time_in_l1_hyperglycemia'] == 30
    assert results['time']['time_in_l2_hyperglycemia'] == 20
