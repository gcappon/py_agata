import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.input_validator import check_same_length_ndarray

import pytest

def test_check_same_length_ndarray():
    """
    Unit test of check_same_length_ndarray function.

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
    # Test not error
    assert check_same_length_ndarray(np.array([0,0]), np.array([1,1]))

    # Test errors
    with pytest.raises(Exception):
        check_same_length_ndarray(np.array([0,0]), np.array([1]))