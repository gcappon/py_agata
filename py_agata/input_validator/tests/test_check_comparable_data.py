import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.input_validator import check_comparable_data

import pytest

def test_check_comparable_data():
    """
    Unit test of check_comparable_data function.

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
    glucose[4:6] = [120, 120]
    glucose[6:8] = [200, 200]
    glucose[8:10] = [260, 260]
    glucose[10] = np.nan
    d = {'t': t, 'glucose': glucose}
    data_1 = pd.DataFrame(data=d)
    d = {'t': t, 'glucose': glucose}
    data_2 = pd.DataFrame(data=d)
    d = {'t': t[0:-1], 'glucose': glucose[0:-1]}
    data_3 = pd.DataFrame(data=d)
    d = {'t': t[1:], 'glucose': glucose[1:]}
    data_4 = pd.DataFrame(data=d)
    t = np.arange(datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 1, 0, 55, 0), timedelta(minutes=5)).astype(
        datetime)
    t[-1] = t[0]
    d = {'t': t, 'glucose': glucose}
    data_5 = pd.DataFrame(data=d)

    # Test not error
    assert check_comparable_data(data_1, data_2)

    # Test errors (different length)
    with pytest.raises(Exception):
        check_comparable_data(data_1, data_3)

    # Test errors (different start)
    with pytest.raises(Exception):
        check_comparable_data(data_3, data_4)

    # Test errors (different end)
    with pytest.raises(Exception):
        check_comparable_data(data_1, data_5)