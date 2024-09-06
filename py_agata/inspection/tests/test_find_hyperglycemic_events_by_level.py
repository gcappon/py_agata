import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.inspection import find_hyperglycemic_events_by_level


def test_find_hyperglycemic_events_by_level():
    """
    Unit test of find_hyperglycemic_events_by_level function.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    None

    See Also
    --------
    None

    Examples
    --------
    None

    References
    ----------
    None
    """
    # Set test data
    t = np.arange(datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 1, 0, 0, 0) + timedelta(minutes=425),
                  timedelta(minutes=5)).astype(datetime)
    glucose = np.ones(shape=(t.shape[0],))*120
    glucose[9:13] = np.ones(shape=(4,))*200
    glucose[29:60] = np.ones(shape=(31,))*200
    glucose[34:40] = np.ones(shape=(6,)) * 300
    glucose[44:50] = np.ones(shape=(6,)) * 300
    glucose[31:33] = [np.nan, np.nan]
    glucose[61:63] = np.ones(shape=(2,)) * 200
    glucose[69:72] = np.ones(shape=(3,)) * 200
    glucose[75:78] = np.ones(shape=(3,)) * 200
    glucose[79:82] = np.ones(shape=(3,)) * 200
    glucose[80] = np.nan

    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    # Hypoglycemic events generated:
    #   - 2000-01-01 @ 00:45 - duration 00:20 (All + L1)
    #   - 2000-01-01 @ 02:45 - duration 02:30 (All)
    #   - 2000-01-01 @ 02:50 - duration 00:30 (L2)
    #   - 2000-01-01 @ 03:40 - duration 00:30 (L2)
    #   - 2000-01-01 @ 05:45 - duration 00:15 (All + L1)
    #   - 2000-01-01 @ 06:15 - duration 00:35 (All + L1)

    # 1: check results structure
    results = find_hyperglycemic_events_by_level(data)
    assert type(results) is dict
    assert type(results['hyper']) is dict
    assert type(results['l1']) is dict
    assert type(results['l2']) is dict
    assert results['hyper']['time_start'].size == results['hyper']['duration'].size
    assert results['hyper']['time_end'].size == results['hyper']['duration'].size
    assert results['l1']['time_start'].size == results['l1']['duration'].size
    assert results['l1']['time_end'].size == results['l1']['duration'].size
    assert results['l2']['time_start'].size == results['l2']['duration'].size
    assert results['l2']['time_end'].size == results['l2']['duration'].size

    # 2. check results (wth events)
    assert results['hyper']['time_start'][0] == datetime(2000, 1, 1, 0, 45, 0)
    assert results['hyper']['time_end'][0] == datetime(2000, 1, 1, 0, 45, 0) + timedelta(minutes=20)
    assert results['hyper']['duration'][0] == 20
    assert results['hyper']['time_start'][1] == datetime(2000, 1, 1, 2, 45, 0)
    assert results['hyper']['time_end'][1] == datetime(2000, 1, 1, 2, 45, 0) + timedelta(minutes=150)
    assert results['hyper']['duration'][1] == 150
    assert results['hyper']['time_start'][2] == datetime(2000, 1, 1, 5, 45, 0)
    assert results['hyper']['time_end'][2] == datetime(2000, 1, 1, 5, 45, 0) + timedelta(minutes=15)
    assert results['hyper']['duration'][2] == 15
    assert results['hyper']['time_start'][3] == datetime(2000, 1, 1, 6, 15, 0)
    assert results['hyper']['time_end'][3] == datetime(2000, 1, 1, 6, 15, 0) + timedelta(minutes=35)
    assert results['hyper']['duration'][3] == 35
    assert results['hyper']['time_start'].size == results['hyper']['duration'].size
    assert results['hyper']['time_end'].size == results['hyper']['duration'].size
    assert results['hyper']['time_start'].size == 4
    assert results['hyper']['mean_duration'] == 55
    assert results['hyper']['events_per_week'] == 96

    assert results['l1']['time_start'][0] == datetime(2000, 1, 1, 0, 45, 0)
    assert results['l1']['time_end'][0] == datetime(2000, 1, 1, 0, 45, 0) + timedelta(minutes=20)
    assert results['l1']['duration'][0] == 20
    assert results['l1']['time_start'][1] == datetime(2000, 1, 1, 5, 45, 0)
    assert results['l1']['time_end'][1] == datetime(2000, 1, 1, 5, 45, 0) + timedelta(minutes=15)
    assert results['l1']['duration'][1] == 15
    assert results['l1']['time_start'][2] == datetime(2000, 1, 1, 6, 15, 0)
    assert results['l1']['time_end'][2] == datetime(2000, 1, 1, 6, 15, 0) + timedelta(minutes=35)
    assert results['l1']['duration'][2] == 35
    assert results['l1']['time_start'].size == results['l1']['duration'].size
    assert results['l1']['time_end'].size == results['l1']['duration'].size
    assert results['l1']['time_start'].size == 3
    assert np.round(results['l1']['mean_duration']*100)/100 == 23.33
    assert results['l1']['events_per_week'] == 72

    assert results['l2']['time_start'][0] == datetime(2000, 1, 1, 2, 50, 0)
    assert results['l2']['time_end'][0] == datetime(2000, 1, 1, 2, 50, 0) + timedelta(minutes=30)
    assert results['l2']['duration'][0] == 30
    assert results['l2']['time_start'][1] == datetime(2000, 1, 1, 3, 40, 0)
    assert results['l2']['time_end'][1] == datetime(2000, 1, 1, 3, 40, 0) + timedelta(minutes=30)
    assert results['l2']['duration'][1] == 30
    assert results['l2']['time_start'].size == results['l2']['duration'].size
    assert results['l2']['time_end'].size == results['l2']['duration'].size
    assert results['l2']['time_start'].size == 2
    assert np.round(results['l2']['mean_duration'] * 100) / 100 == 30
    assert results['l2']['events_per_week'] == 48

    # 3. check results (without events)
    data.glucose = np.ones(shape=(data.glucose.values.size,))*120
    results = find_hyperglycemic_events_by_level(data)
    assert results['hyper']['time_start'].size == results['hyper']['duration'].size
    assert results['hyper']['time_end'].size == results['hyper']['duration'].size
    assert results['hyper']['time_start'].size == 0
    assert np.isnan(results['hyper']['mean_duration'])
    assert results['hyper']['events_per_week'] == 0

    assert results['l1']['time_start'].size == results['l1']['duration'].size
    assert results['l1']['time_end'].size == results['l1']['duration'].size
    assert results['l1']['time_start'].size == 0
    assert np.isnan(results['l1']['mean_duration'])
    assert results['l1']['events_per_week'] == 0

    assert results['l2']['time_start'].size == results['l2']['duration'].size
    assert results['l2']['time_end'].size == results['l2']['duration'].size
    assert results['l2']['time_start'].size == 0
    assert np.isnan(results['l2']['mean_duration'])
    assert results['l2']['events_per_week'] == 0

    # Set empty data
    d = {'t': [], 'glucose': []}
    data = pd.DataFrame(data=d)

    # 5. Test with empty data
    results = find_hyperglycemic_events_by_level(data)
    assert results['hyper']['time_start'].size == results['hyper']['duration'].size
    assert results['hyper']['time_end'].size == results['hyper']['duration'].size
    assert results['hyper']['time_start'].size == 0
    assert np.isnan(results['hyper']['mean_duration'])
    assert np.isnan(results['hyper']['events_per_week'])

    assert results['l1']['time_start'].size == results['l1']['duration'].size
    assert results['l1']['time_end'].size == results['l1']['duration'].size
    assert results['l1']['time_start'].size == 0
    assert np.isnan(results['l1']['mean_duration'])
    assert np.isnan(results['l1']['events_per_week'])

    assert results['l2']['time_start'].size == results['l2']['duration'].size
    assert results['l2']['time_end'].size == results['l2']['duration'].size
    assert results['l2']['time_start'].size == 0
    assert np.isnan(results['l2']['mean_duration'])
    assert np.isnan(results['l2']['events_per_week'])
