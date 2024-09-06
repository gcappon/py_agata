import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.processing import retime_glucose


def test_retime_glucose():
    """
    Unit test of retime_glucose function.

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
    t = np.arange(datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 1, 0, 0, 0)+timedelta(minutes=30), timedelta(minutes=5)).astype(
        datetime)
    t[0] = t[0].replace(second=17)
    t[1] = t[1].replace(second=33)
    t[2] = t[2].replace(second=58)
    t[3] = t[3].replace(second=58)
    t[4] = t[4].replace(second=10)
    t[5] = t[5].replace(second=59)

    glucose = np.ones(shape=(t.size,))
    glucose[0] = 40
    glucose[1] = 50
    glucose[2:4] = [np.nan, np.nan]
    glucose[4:6] = [120, 120]

    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    #Tests
    results = retime_glucose(data, 5)

    # 1. check retime (timestep = 5)

    assert results.glucose.values[0] == 40
    assert results.glucose.values[1] == 50
    assert np.isnan(results.glucose.values[2])
    assert np.isnan(results.glucose.values[3])
    assert results.glucose.values[4] == 120
    assert results.glucose.values[5] == 120

    assert pd.to_datetime(results.t.values[0]).to_pydatetime().second == 0
    assert pd.to_datetime(results.t.values[1]).to_pydatetime().second == 0
    assert pd.to_datetime(results.t.values[2]).to_pydatetime().second == 0
    assert pd.to_datetime(results.t.values[3]).to_pydatetime().second == 0
    assert pd.to_datetime(results.t.values[4]).to_pydatetime().second == 0
    assert pd.to_datetime(results.t.values[5]).to_pydatetime().second == 0

    assert pd.to_datetime(results.t.values[0]).to_pydatetime().minute == 0
    assert pd.to_datetime(results.t.values[1]).to_pydatetime().minute == 5
    assert pd.to_datetime(results.t.values[2]).to_pydatetime().minute == 10
    assert pd.to_datetime(results.t.values[3]).to_pydatetime().minute == 15
    assert pd.to_datetime(results.t.values[4]).to_pydatetime().minute == 20
    assert pd.to_datetime(results.t.values[5]).to_pydatetime().minute == 25

    # 2. check retime (timestep = 10)
    results = retime_glucose(data, 10)

    assert results.glucose.values[0] == 40
    assert results.glucose.values[1] == 50
    assert results.glucose.values[2] == 120

    assert pd.to_datetime(results.t.values[0]).to_pydatetime().second == 0
    assert pd.to_datetime(results.t.values[1]).to_pydatetime().second == 0
    assert pd.to_datetime(results.t.values[2]).to_pydatetime().second == 0

    assert pd.to_datetime(results.t.values[0]).to_pydatetime().minute == 0
    assert pd.to_datetime(results.t.values[1]).to_pydatetime().minute == 10
    assert pd.to_datetime(results.t.values[2]).to_pydatetime().minute == 20

    # 3. check retime (timestep = 12)
    results = retime_glucose(data, 12)

    assert results.glucose.values[0] == 45
    assert np.isnan(results.glucose.values[1])
    assert results.glucose.values[2] == 120

    assert pd.to_datetime(results.t.values[0]).to_pydatetime().second == 0
    assert pd.to_datetime(results.t.values[1]).to_pydatetime().second == 0
    assert pd.to_datetime(results.t.values[2]).to_pydatetime().second == 0

    assert pd.to_datetime(results.t.values[0]).to_pydatetime().minute == 0
    assert pd.to_datetime(results.t.values[1]).to_pydatetime().minute == 12
    assert pd.to_datetime(results.t.values[2]).to_pydatetime().minute == 24
