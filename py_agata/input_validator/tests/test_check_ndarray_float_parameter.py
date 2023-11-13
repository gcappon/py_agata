import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.input_validator import check_ndarray_float_parameter

import pytest

def test_check_ndarray_float_parameter():
    """
    Unit test of check_ndarray_float_parameter function.

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
    assert check_ndarray_float_parameter(np.array([50., 40.]))
    assert check_ndarray_float_parameter(np.array([]))

    # Test errors
    with pytest.raises(Exception):
        check_ndarray_float_parameter(50)

    # Test errors
    with pytest.raises(Exception):
        check_ndarray_float_parameter(np.array([50]))