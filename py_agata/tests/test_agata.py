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
    #Tests
    agata = Agata(glycemic_target='diabetes')

    assert agata.glycemic_target == 'diabetes'
