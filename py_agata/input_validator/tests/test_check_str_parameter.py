import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.input_validator import check_str_parameter

import pytest

def test_check_str_parameter():
    """
    Unit test of check_str_parameter function.

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
    assert check_str_parameter("ciao")

    # Test errors
    with pytest.raises(Exception):
        check_str_parameter(50)