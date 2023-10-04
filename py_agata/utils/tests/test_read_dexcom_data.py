import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta
import os
from py_agata.utils import read_dexcom_data


def test_read_dexcom_data():
    """
    Unit test of read_dexcom_data function.

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

    file = os.path.join(os.path.abspath(''),'example','data','dexcom_example.xlsx')
    # d = {'t': time_range, 'glucose': data.glucose.values}
    # data = pd.DataFrame(data=d)

    #Tests
    data = read_dexcom_data(file)

    assert type(data) is pd.DataFrame
    assert 't' in data.columns
    assert 'glucose' in data.columns

    assert data.glucose.values.size == 3814
    assert np.all(np.isnan(data.glucose.values[8:10]))
    assert np.all(np.isnan(data.glucose.values[12:18]))
    assert data.glucose.values[10] == 401
    assert data.glucose.values[11] == 39
