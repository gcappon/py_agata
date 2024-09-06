import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta
import os
from py_agata.utils import read_freestyle_libre_data


def test_read_freestyle_libre_data():
    """
    Unit test of read_freestyle_libre_data function.

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

    file = os.path.join(os.path.abspath(''),'example','data','freestyle_libre_example.xlsx')
    # d = {'t': time_range, 'glucose': data.glucose.values}
    # data = pd.DataFrame(data=d)

    #Tests
    data = read_freestyle_libre_data(file)

    assert type(data) is pd.DataFrame
    assert 't' in data.columns
    assert 'glucose' in data.columns

    assert data.glucose.values.size == 1440
    assert np.isnan(data.glucose.values[0])
    assert data.glucose.values[1] == 432
