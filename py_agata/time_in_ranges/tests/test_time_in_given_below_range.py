import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.time_in_ranges import time_in_given_below_range


def test_time_in_given_below_range():
    """
    Unit test of time_in_given_below_range function.

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
    t = np.arange(datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 1, 0, 55, 0), timedelta(minutes=5)).astype(
        datetime)
    glucose = np.zeros(shape=(t.shape[0],))
    glucose[0] = 40
    glucose[1:3] = [60, 60]
    glucose[3] = 80
    glucose[4:6] = [120, 130]
    glucose[6:8] = [200, 200]
    glucose[8:10] = [260, 260]
    glucose[10] = np.nan
    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    #Tests
    assert np.isnan(time_in_given_below_range(data,60, include_th=False)) == False
    assert time_in_given_below_range(data,60, include_th=False) == 10
    assert time_in_given_below_range(data,60, include_th=True) == 30

    # Set empty data
    t = np.arange(datetime(2000, 1, 1, 1, 0, 0), datetime(2000, 1, 1, 1, 55, 0), timedelta(minutes=5)).astype(
        datetime)
    glucose = np.zeros(shape=(t.shape[0],))
    glucose[0] = np.nan
    glucose[1:3] = [np.nan, np.nan]
    glucose[3] = np.nan
    glucose[4:6] = [np.nan, np.nan]
    glucose[6:8] = [np.nan, np.nan]
    glucose[8:10] = [np.nan, np.nan]
    glucose[10] = np.nan
    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    # Tests
    assert np.isnan(time_in_given_below_range(data, th=60))