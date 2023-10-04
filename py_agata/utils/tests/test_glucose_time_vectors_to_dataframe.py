import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.utils import glucose_time_vectors_to_dataframe


def test_glucose_time_vectors_to_dataframe():
    """
    Unit test of glucose_time_vectors_to_dataframe function.

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
    t = np.arange(datetime(2000, 1, 1, 0, 30, 0), datetime(2000, 1, 1, 0, 45, 0), timedelta(minutes=5)).astype(
        datetime)
    glucose = np.zeros(shape=(t.shape[0],))
    glucose[0] = np.nan
    glucose[1:3] = [100, 200]

    #Tests
    results = glucose_time_vectors_to_dataframe(glucose=glucose, t=t)
    assert type(results) is pd.DataFrame
    assert 't' in results.columns
    assert 'glucose' in results.columns

    assert np.isnan(results.glucose.values[0])
    assert results.glucose.values[1] == 100
    assert results.glucose.values[2] == 200
    assert results.t.to_dict()[0].to_pydatetime() == t[0]
    assert results.t.to_dict()[1].to_pydatetime() == t[1]
    assert results.t.to_dict()[2].to_pydatetime() == t[2]

