import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.utils import glucose_vector_to_dataframe


def test_glucose_vector_to_dataframe():
    """
    Unit test of glucose_vector_to_dataframe function.

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
    t1 = np.arange(datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 1, 0, 15, 0), timedelta(minutes=5)).astype(
        datetime)
    t2 = np.arange(datetime(2000, 1, 1, 1, 0, 0), datetime(2000, 1, 1, 1, 15, 0), timedelta(minutes=5)).astype(
        datetime)
    glucose = np.zeros(shape=(t1.shape[0],))
    glucose[0] = np.nan
    glucose[1:3] = [100, 200]

    #Tests
    results = glucose_vector_to_dataframe(glucose=glucose, sample_time=5)
    assert type(results) is pd.DataFrame
    assert 't' in results.columns
    assert 'glucose' in results.columns

    assert np.isnan(results.glucose.values[0])
    assert results.glucose.values[1] == 100
    assert results.glucose.values[2] == 200
    assert results.t.to_dict()[0].to_pydatetime() == t1[0]
    assert results.t.to_dict()[1].to_pydatetime() == t1[1]
    assert results.t.to_dict()[2].to_pydatetime() == t1[2]

    results = glucose_vector_to_dataframe(glucose=glucose, sample_time=5, start_time=t2[0])
    assert type(results) is pd.DataFrame
    assert 't' in results.columns
    assert 'glucose' in results.columns

    assert np.isnan(results.glucose.values[0])
    assert results.glucose.values[1] == 100
    assert results.glucose.values[2] == 200
    assert results.t.to_dict()[0].to_pydatetime() == t2[0]
    assert results.t.to_dict()[1].to_pydatetime() == t2[1]
    assert results.t.to_dict()[2].to_pydatetime() == t2[2]

