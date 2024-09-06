import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.inspection import find_nan_islands


def test_find_nan_islands():
    """
    Unit test of find_nan_islands function.

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
    t = np.arange(datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 1, 0, 0, 0)+timedelta(minutes=120), timedelta(minutes=5)).astype(
        datetime)
    glucose = np.zeros(shape=(t.size,))
    glucose[0] = 40
    glucose[1:3] = [50, 50]
    glucose[3] = 80
    glucose[4:9] = [np.nan, np.nan, np.nan, np.nan, np.nan]
    glucose[10:12] = [200, 200]
    glucose[12:15] = [260, 260, 260]
    glucose[15:17] = [np.nan, np.nan]
    glucose[17] = 180
    glucose[18] = np.nan
    glucose[19] = 140
    glucose[20:24] = [np.nan, np.nan, np.nan, np.nan]
    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    # 1. check NaN presence
    short_nan, long_nan, nan_start, nan_end = find_nan_islands(data,0)
    assert np.any(np.isnan(short_nan)) == False
    assert np.any(np.isnan(long_nan)) == False
    assert np.any(np.isnan(nan_start)) == False
    assert np.any(np.isnan(nan_end)) == False

    # 2. check nan_start calculation
    short_nan, long_nan, nan_start, nan_end = find_nan_islands(data, 2)
    assert nan_start[0] == 4
    assert nan_start[1] == 15
    assert nan_start[2] == 18
    assert nan_start[3] == 20
    assert nan_start.size == 4

    # 3. check nan_end calculation
    assert nan_end[0] == 8
    assert nan_end[1] == 16
    assert nan_end[2] == 18
    assert nan_end[3] == 23
    assert nan_end.size == 4


    # 4a. check shortNan calculation (TH = 3)
    short_nan, long_nan, nan_start, nan_end = find_nan_islands(data, 3)
    assert short_nan[0] == 15
    assert short_nan[1] == 16
    assert short_nan[2] == 18
    assert short_nan.size == 3

    # 4b. check short_nan calculation (TH = 10000)
    short_nan, long_nan, nan_start, nan_end = find_nan_islands(data, 10000)
    assert short_nan.size == 12

    # 4c. check short_nan calculation (TH = 1)
    short_nan, long_nan, nan_start, nan_end = find_nan_islands(data, 1)
    assert short_nan.size == 0

    # 5a. check long_nan calculation (TH = 4)
    short_nan, long_nan, nan_start, nan_end = find_nan_islands(data, 4)
    assert long_nan[0] == 4
    assert long_nan[1] == 5
    assert long_nan[2] == 6
    assert long_nan[3] == 7
    assert long_nan[4] == 8
    assert long_nan[5] == 20
    assert long_nan[6] == 21
    assert long_nan[7] == 22
    assert long_nan[8] == 23
    assert long_nan.size == 9

    # 5b. check longNan calculation (TH = 10000)
    short_nan, long_nan, nan_start, nan_end = find_nan_islands(data, 10000)
    assert long_nan.size == 0

    # 5c. check longNan calculation (TH = 1)
    short_nan, long_nan, nan_start, nan_end = find_nan_islands(data, 1)
    assert long_nan.size == 12

    # Set test data
    t = np.arange(datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 1, 0, 0, 0)+timedelta(minutes=15), timedelta(minutes=5)).astype(
        datetime)
    glucose = np.zeros(shape=(t.size,))
    glucose[0] = 40
    glucose[1:3] = [50, 50]
    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    # 6. check what happens when no nans are present
    short_nan, long_nan, nan_start, nan_end = find_nan_islands(data, 1)
    assert short_nan.size == 0
    assert long_nan.size == 0
    assert nan_start.size == 0
    assert nan_end.size == 0

    d = {'t': t[0:2], 'glucose': [120, np.nan]}
    data = pd.DataFrame(data=d)

    # 7a. check what happens when 1 nan is present
    short_nan, long_nan, nan_start, nan_end = find_nan_islands(data, 1)
    assert short_nan.size == 0
    assert long_nan.size == 1
    assert nan_start.size == 1
    assert nan_end.size == 1

    # 7b. check what happens when 1 nan is present
    short_nan, long_nan, nan_start, nan_end = find_nan_islands(data, 2)
    assert short_nan.size == 1
    assert long_nan.size == 0
    assert nan_start.size == 1
    assert nan_end.size == 1