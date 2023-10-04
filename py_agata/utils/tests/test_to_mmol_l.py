import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.utils import to_mmol_l


def test_to_mmol_l():
    """
    Unit test of to_mmol_l function.

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
    glucose[1:3] = [1801.8, 3603.6]
    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    #Tests
    results = to_mmol_l(data)
    assert type(results) is pd.DataFrame
    assert 't' in results.columns
    assert 'glucose' in results.columns
    assert np.isnan(results.glucose.values[0])
    assert np.round(results.glucose.values[1]*10)/10 == 100
    assert np.round(results.glucose.values[2]*10)/10 == 200

    results = to_mmol_l(np.array([1801.8, np.nan]))
    assert type(results) is np.ndarray
    assert np.round(results[0]*10)/10 == 100
    assert np.isnan(results[1])

    results = to_mmol_l(1801.8)
    assert type(results) is float
    assert np.round(results * 10) / 10 == 100

    results = to_mmol_l(np.nan)
    assert type(results) is float
    assert np.isnan(results)
