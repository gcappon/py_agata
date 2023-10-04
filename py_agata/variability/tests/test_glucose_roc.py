import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.variability import glucose_roc


def test_glucose_roc():
    """
    Unit test of glucose_roc function.

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

    result = glucose_roc(data)

    # 1. check if results is a pd.Dataframe
    assert type(result) is pd.DataFrame

    # 2. check the columns of results
    assert 't' in result.columns
    assert 'glucose_roc' in result.columns

    # 3: check that the first and last timestamps coincide with the one of the input
    assert result.t.values[0] == data.t.values[0]
    assert result.t.values[-1] == data.t.values[-1]

    # 4. check that the length coincides with the one of the input
    assert result.glucose_roc.values.size == data.glucose.values.size

    # 5. check results calculation
    assert np.isnan(result.glucose_roc.values[0])
    assert np.isnan(result.glucose_roc.values[1])
    assert np.isnan(result.glucose_roc.values[2])
    assert np.round(result.glucose_roc.values[3]*100)/100 == 2.67
    assert np.round(result.glucose_roc.values[4] * 100) / 100 == 4.67
    assert np.round(result.glucose_roc.values[5] * 100) / 100 == 4.67
    assert np.round(result.glucose_roc.values[6] * 100) / 100 == 8
    assert np.round(result.glucose_roc.values[7] * 100) / 100 == 5.33
    assert np.round(result.glucose_roc.values[8] * 100) / 100 == 9.33
    assert np.round(result.glucose_roc.values[9] * 100) / 100 == 4
    assert (np.isnan(result.glucose_roc.values[10]))