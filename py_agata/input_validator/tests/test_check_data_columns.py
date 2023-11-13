import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.input_validator import check_data_columns

import pytest

def test_check_data_columns():
    """
    Unit test of check_data_columns function.

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
    data = pd.DataFrame(data=d)

    # Test not error
    assert check_data_columns(data)

    d = {'t': t, 'g': glucose}
    data = pd.DataFrame(data=d)

    # Test errors
    with pytest.raises(Exception):
        check_data_columns(data)

    d = {'time': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    # Test errors
    with pytest.raises(Exception):
        check_data_columns(data)