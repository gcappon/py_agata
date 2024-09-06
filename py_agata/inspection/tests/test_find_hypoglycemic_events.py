import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.inspection import find_hypoglycemic_events


def test_find_hypoglycemic_events():
    """
    Unit test of find_hypoglycemic_events function.

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
    glucose[9:13] = np.ones(shape=(4,))*50
    glucose[29:60] = np.ones(shape=(31,))*50
    glucose[31:33] = [np.nan, np.nan]
    glucose[61:63] = np.ones(shape=(2,)) * 50
    glucose[69:72] = np.ones(shape=(3,)) * 50
    glucose[75:78] = np.ones(shape=(3,)) * 50
    glucose[79:82] = np.ones(shape=(3,)) * 50
    glucose[80] = np.nan

    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    # Hypoglycemic events generated:
    #   - 2000-01-01 @ 00:45 - duration 00:20 (normal hypoglycemic event)
    #   - 2000-01-01 @ 02:45 - duration 02:30 (extended hypoglycemic event)
    #   - 2000-01-01 @ 05:45 - duration 00:15 (normal hypoglycemic event)
    #   - 2000-01-01 @ 06:15 - duration 00:35 (normal hypoglycemic event)

    # 1: check results structure
    results = find_hypoglycemic_events(data)
    assert type(results) is dict
    assert results['time_start'].size == results['duration'].size
    assert results['time_end'].size == results['duration'].size

    # 2. check results (wth events)
    assert results['time_start'][0] == datetime(2000, 1, 1, 0, 45, 0)
    assert results['time_end'][0] == datetime(2000, 1, 1, 0, 45, 0) + timedelta(minutes=20)
    assert results['duration'][0] == 20
    assert results['time_start'][1] == datetime(2000, 1, 1, 2, 45, 0)
    assert results['time_end'][1] == datetime(2000, 1, 1, 2, 45, 0) + timedelta(minutes=150)
    assert results['duration'][1] == 150
    assert results['time_start'][2] == datetime(2000, 1, 1, 5, 45, 0)
    assert results['time_end'][2] == datetime(2000, 1, 1, 5, 45, 0) + timedelta(minutes=15)
    assert results['duration'][2] == 15
    assert results['time_start'][3] == datetime(2000, 1, 1, 6, 15, 0)
    assert results['time_end'][3] == datetime(2000, 1, 1, 6, 15, 0) + timedelta(minutes=35)
    assert results['duration'][3] == 35
    assert results['time_start'].size == results['duration'].size
    assert results['time_end'].size == results['duration'].size
    assert results['time_start'].size == 4
    assert results['mean_duration'] == 55
    assert results['events_per_week'] == 96

    # 3. check results (with custom threshold)
    results = find_hypoglycemic_events(data, th=45.)
    assert results['time_start'].size == results['duration'].size
    assert results['time_end'].size == results['duration'].size
    assert results['time_start'].size == 0
    assert np.isnan(results['mean_duration'])
    assert results['events_per_week'] == 0

    # 4. check results (without events)
    data.glucose = np.ones(shape=(data.glucose.values.size,))*120
    results = find_hypoglycemic_events(data)
    assert results['time_start'].size == results['duration'].size
    assert results['time_end'].size == results['duration'].size
    assert results['time_start'].size == 0
    assert np.isnan(results['mean_duration'])
    assert results['events_per_week'] == 0


    # Set empty data
    d = {'t': [], 'glucose': []}
    data = pd.DataFrame(data=d)

    # 5. Test with empty data
    results = find_hypoglycemic_events(data)
    assert results['time_start'].size == results['duration'].size
    assert results['time_end'].size == results['duration'].size
    assert results['time_start'].size == 0
    assert np.isnan(results['mean_duration'])
    assert np.isnan(results['events_per_week'])

    # Set test data
    t = np.arange(datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 1, 0, 0, 0) + timedelta(minutes=435),
                  timedelta(minutes=5)).astype(datetime)
    glucose = np.ones(shape=(t.shape[0],)) * 120
    glucose[9:13] = np.ones(shape=(4,)) * 50
    glucose[29:60] = np.ones(shape=(31,)) * 50
    glucose[31:33] = [np.nan, np.nan]
    glucose[61:63] = np.ones(shape=(2,)) * 100
    glucose[69:72] = np.ones(shape=(3,)) * 100
    glucose[75:78] = np.ones(shape=(3,)) * 100
    glucose[79:82] = np.ones(shape=(3,)) * 100
    glucose[84:87] = np.ones(shape=(3,)) * 50
    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    # 6. Test last entry
    results = find_hypoglycemic_events(data)
    assert results['time_start'][-1] == datetime(2000, 1, 1, 7, 0, 0)
    assert results['time_end'][-1] == datetime(2000, 1, 1, 7, 0, 0) + timedelta(minutes=15)
    assert results['duration'][-1] == 15

    # Set test data
    t = np.arange(datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 1, 0, 0, 0) + timedelta(minutes=440),
                  timedelta(minutes=5)).astype(datetime)
    glucose = np.ones(shape=(t.shape[0],)) * 120
    glucose[9:13] = np.ones(shape=(4,)) * 50
    glucose[29:60] = np.ones(shape=(31,)) * 50
    glucose[31:33] = [np.nan, np.nan]
    glucose[61:63] = np.ones(shape=(2,)) * 100
    glucose[69:72] = np.ones(shape=(3,)) * 100
    glucose[75:78] = np.ones(shape=(3,)) * 100
    glucose[79:82] = np.ones(shape=(3,)) * 100
    glucose[84:87] = np.ones(shape=(3,)) * 50
    glucose[87] = 100
    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    # 6b. Test last entry
    results = find_hypoglycemic_events(data)
    assert results['time_start'][-1] == datetime(2000, 1, 1, 7, 0, 0)
    assert results['time_end'][-1] == datetime(2000, 1, 1, 7, 0, 0) + timedelta(minutes=15)
    assert results['duration'][-1] == 15