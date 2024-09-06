import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.inspection import number_days_of_observation


def test_number_days_of_observation():
    """
    Unit test of number_days_of_observation function.

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
    t = np.arange(datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 1, 0, 0, 0) + timedelta(minutes=425),
                  timedelta(minutes=5)).astype(datetime)
    glucose = np.ones(shape=(t.shape[0],))*120
    glucose[80] = np.nan
    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    #Tests
    assert np.isnan(number_days_of_observation(data)) == False
    assert np.round(number_days_of_observation(data)*100)/100 == 0.29

    # Set empty data
    d = {'t': [], 'glucose': []}
    data = pd.DataFrame(data=d)

    # Tests
    assert np.isnan(number_days_of_observation(data))
