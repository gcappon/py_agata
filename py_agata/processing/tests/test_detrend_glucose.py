import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.processing import detrend_glucose


def test_detrend_glucose():
    """
    Unit test of detrend_glucose function.

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
    t = np.arange(datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 1, 0, 0, 0)+timedelta(minutes=100), timedelta(minutes=5)).astype(
        datetime)
    glucose = np.arange(100,200,5)*1.0
    glucose[0:3] = [np.nan, np.nan, np.nan]
    glucose[9:20] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    #Tests
    results = detrend_glucose(data)

    assert (results.glucose[3] == 100)
    assert (results.glucose[4] == 100)
    assert (results.glucose[5] == 100)
    assert (results.glucose[6] == 100)
    assert (results.glucose[7] == 100)
    assert (results.glucose[8] == 100)

    assert np.all(np.isnan(data.glucose.values[0:3]))
    assert np.all(np.isnan(data.glucose.values[9:20]))

    # Set test data
    d = {'t': t[0], 'glucose': [100]}
    data = pd.DataFrame(data=d)

    # Tests
    results = detrend_glucose(data)
    assert (results.glucose[0] == 100)