import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.processing import impute_glucose


def test_impute_glucose():
    """
    Unit test of impute_glucose function.

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
    t = np.arange(datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 1, 0, 0, 0)+timedelta(minutes=125), timedelta(minutes=5)).astype(
        datetime)
    glucose = np.ones(shape=(t.size,))*120
    glucose[1:3] = [np.nan, np.nan]
    glucose[9:20] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    glucose[21] = np.nan
    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    #Tests
    results = impute_glucose(data, 15)

    assert results.glucose.values.size == data.glucose.values.size
    assert (results.glucose[1] == 120)
    assert (results.glucose[2] == 120)
    assert (results.glucose[21] == 120)
    assert np.all(np.isnan(data.glucose.values[9:20]))