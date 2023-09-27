import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.py_agata import Agata


def test_agata_init():
    """
    Unit test of Agata constructor.

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
    glucose[1:3] = [60, 60]
    glucose[3] = 80
    glucose[4:6] = [120, 130]
    glucose[6:8] = [200, 200]
    glucose[8:10] = [260, 260]
    glucose[10] = np.nan
    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    #Tests
    agata = Agata(data=data, glycemic_target='diabetes')

    assert agata.glycemic_target == 'diabetes'
    assert agata.data is data
