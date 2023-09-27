import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.risk import dynamic_risk


def test_dynamic_risk():
    """
    Unit test of dynamic_risk function.

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
    t = np.arange(datetime(2000, 1, 1, 1, 0, 0), datetime(2000, 1, 1, 2, 5, 0), timedelta(minutes=5)).astype(
        datetime)
    glucose = np.zeros(shape=(t.shape[0],))
    glucose[0] = 40
    glucose[1:3] = [50, 50]
    glucose[3] = 80
    glucose[4:6] = [120, 120]
    glucose[6:8] = [200, 200]
    glucose[8:10] = [260, 260]
    glucose[10] = np.nan
    glucose[11] = 260
    glucose[12] = 265

    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    #Tests

    # 1. check the first value of result (independent on any parameter)
    assert np.round(dynamic_risk(data)[0]*100)/100 == -36.42

    # 2. test the function with different amplification_function parameter
    results = dynamic_risk(data, amplification_function='exp')
    assert np.round(results[0]*100)/100 == -36.42
    assert np.round(np.nanmean(results)/10)*10 == 5630
    assert np.round(results[-1]*100)/100 == 45.87

    # 3. test the function with different maximum_amplification parameter
    results = dynamic_risk(data, maximum_amplification=1.2)
    assert np.round(results[0]*100)/100 == -36.42
    assert np.round(np.nanmean(results)*100)/100 == 3.07
    assert np.round(results[-1]*100)/100 == 28.44

    # 4. test the function with different amplification_rapidity parameter
    results = dynamic_risk(data, amplification_rapidity=2.5)
    assert np.round(results[0]*100)/100 == -36.42
    assert np.round(np.nanmean(results)*100)/100 == 8.23
    assert np.round(results[-1]*100)/100 == 37.93

    # 5. test the function with different maximum_damping parameter
    results = dynamic_risk(data, maximum_damping=1.8)
    assert np.round(results[0]*100)/100 == -36.42
    assert np.round(np.nanmean(results)*100)/100 == 12.71
    assert np.round(results[-1]*100)/100 == 118.7

    # 6. test the function with a combination of parameters
    results = dynamic_risk(data, amplification_function='tanh', maximum_amplification=2.2, amplification_rapidity=1.7, maximum_damping=1.1)
    assert np.round(results[0]*100)/100 == -36.42
    assert np.round(np.nanmean(results)*100)/100 == 4.59
    assert np.round(results[-1]*100)/100 == 22.12

    # 7. test different order of parameters
    results = dynamic_risk(data, amplification_rapidity=1.7, maximum_amplification=2.2, maximum_damping=1.1,amplification_function='tanh')
    assert np.round(results[0]*100)/100 == -36.42
    assert np.round(np.nanmean(results)*100)/100 == 4.59
    assert np.round(results[-1]*100)/100 == 22.12

    # Set empty test data
    t = []
    glucose = []
    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    # 8. return nan if data is empty
    assert np.isnan(dynamic_risk(data))
