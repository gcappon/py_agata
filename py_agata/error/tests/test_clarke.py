import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.error import clarke


def test_clarke():
    """
    Unit test of clarke function.

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
    glucose[1:3] = [50, 50]
    glucose[3] = 80
    glucose[4:6] = [180, 180]
    glucose[6:8] = [170, 170]
    glucose[8:10] = [260, 100]
    glucose[10] = np.nan
    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    t = np.arange(datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 1, 0, 55, 0), timedelta(minutes=5)).astype(
        datetime)
    glucose = np.zeros(shape=(t.shape[0],))
    glucose[0] = 80
    glucose[1:3] = [80, 80]
    glucose[3] = 70
    glucose[4:6] = [130, 130]
    glucose[6:8] = [np.nan, np.nan]
    glucose[8:10] = [60, 220]
    glucose[10] = 60
    d = {'t': t, 'glucose': glucose}
    data_hat = pd.DataFrame(data=d)

    #Tests
    results = clarke(data,data_hat)
    assert type(results) is dict
    assert results["a"] == 12.5
    assert results["b"] == 25
    assert results["c"] == 12.5
    assert results["d"] == 37.5
    assert results["e"] == 12.5

    d = {'t': [], 'glucose': []}
    data_hat = pd.DataFrame(data=d)
    d = {'t': [], 'glucose': []}
    data = pd.DataFrame(data=d)

    # Tests
    assert np.isnan(clarke(data, data_hat)["a"])
    assert np.isnan(clarke(data, data_hat)["b"])
    assert np.isnan(clarke(data, data_hat)["c"])
    assert np.isnan(clarke(data, data_hat)["d"])
    assert np.isnan(clarke(data, data_hat)["e"])

    d = {'t': [t[0]], 'glucose': [np.nan]}
    data_hat = pd.DataFrame(data=d)
    d = {'t': [t[0]], 'glucose': [120]}
    data = pd.DataFrame(data=d)

    # Tests
    assert np.isnan(clarke(data, data_hat)["a"])
    assert np.isnan(clarke(data, data_hat)["b"])
    assert np.isnan(clarke(data, data_hat)["c"])
    assert np.isnan(clarke(data, data_hat)["d"])
    assert np.isnan(clarke(data, data_hat)["e"])