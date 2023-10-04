import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.utils import to_mg_dl


def test_to_mg_dl():
    """
    Unit test of to_mg_dl function.

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
    t = np.arange(datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 1, 0, 15, 0), timedelta(minutes=5)).astype(
        datetime)
    glucose = np.zeros(shape=(t.shape[0],))
    glucose[0] = np.nan
    glucose[1:3] = [100, 200]
    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    #Tests
    results = to_mg_dl(data)
    assert type(results) is pd.DataFrame
    assert 't' in results.columns
    assert 'glucose' in results.columns
    assert np.isnan(results.glucose.values[0])
    assert np.round(results.glucose.values[1]*10)/10 == 1801.8
    assert np.round(results.glucose.values[2]*10)/10 == 3603.6

    results = to_mg_dl(np.array([100, np.nan]))
    assert type(results) is np.ndarray
    assert np.round(results[0] * 10) / 10 == 1801.8
    assert np.isnan(results[1])

    results = to_mg_dl(100)
    assert type(results) is float
    assert np.round(results * 10) / 10 == 1801.8

    results = to_mg_dl(np.nan)
    assert type(results) is float
    assert np.isnan(results)