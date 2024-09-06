import numpy as np
from datetime import datetime
import pandas as pd


def check_dataframe(data):
    """
    Checks that data is a pd.Dataframe

    Parameters
    ----------
    data: pd.DataFrame
        Input data

    Returns
    -------
    is_ok: bool
        True if given data is ok

    Raises
    ------
    exception: exception
        If data is not valid

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
    if not type(data) is pd.DataFrame:
        raise Exception("`data` is not a pd.Dataframe")

    return True


def check_data_columns(data):
    """
    Checks that data has a `glucose` and a `t` column

    Parameters
    ----------
    data: pd.DataFrame
        Input data

    Returns
    -------
    is_ok: bool
        True if given data is ok

    Raises
    ------
    exception: exception
        If data is not valid

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
    if not 't' in data.columns.values:
        raise Exception("`data` must have a `t` column")
    if not 'glucose' in data.columns.values:
        raise Exception("`data` must have a `glucose` column")

    return True


def check_homogeneous_timegrid(data):
    """
    Checks that data has a homogeneous timegrid

    Parameters
    ----------
    data: pd.DataFrame
        Input data

    Returns
    -------
    is_ok: bool
        True if given data is ok

    Raises
    ------
    exception: exception
        If data is not valid

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
    d = np.diff(data.t)
    if d.size == 0:
        return True

    if not np.all(d == d[0]):
        raise Exception("`data` has not an homogeneous timegrid")

    return True


def check_float_parameter(parameter):
    """
    Checks that parameter is a float

    Parameters
    ----------
    parameter: float
        Input parameter

    Returns
    -------
    is_ok: bool
        True if given parameter is ok

    Raises
    ------
    exception: exception
        If parameter is not valid

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
    if not type(parameter) is float:
        raise Exception("`parameter` is not a float")

    return True


def check_int_parameter(parameter):
    """
    Checks that parameter is a int

    Parameters
    ----------
    parameter: int
        Input parameter

    Returns
    -------
    is_ok: bool
        True if given parameter is ok

    Raises
    ------
    exception: exception
        If parameter is not valid

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
    if not type(parameter) is int:
        raise Exception("`parameter` is not an int")

    return True


def check_datetime_parameter(parameter):
    """
    Checks that parameter is a datetime

    Parameters
    ----------
    parameter: datetime
        Input parameter

    Returns
    -------
    is_ok: bool
        True if given parameter is ok

    Raises
    ------
    exception: exception
        If parameter is not valid

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
    if not type(parameter) is datetime:
        raise Exception("`parameter` is not a datetime")

    return True


def check_str_parameter(parameter):
    """
    Checks that parameter is a str

    Parameters
    ----------
    parameter: str
        Input parameter

    Returns
    -------
    is_ok: bool
        True if given parameter is ok

    Raises
    ------
    exception: exception
        If parameter is not valid

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
    if not type(parameter) is str:
        raise Exception("`parameter` is not a str")

    return True


def check_ndarray_float_parameter(parameter):
    """
    Checks that parameter is a np.ndarray of flaot

    Parameters
    ----------
    parameter: np.ndarray of float
        Input parameter

    Returns
    -------
    is_ok: bool
        True if given parameter is ok

    Raises
    ------
    exception: exception
        If parameter is not valid

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
    if not type(parameter) is np.ndarray:
        raise Exception("`parameter` is not a np.ndarray")

    if parameter.size == 0:
        return True

    if not type(parameter[0]) is np.float_:
        raise Exception("`parameter` must contain float values")

    return True


def check_ndarray_datetime_parameter(parameter):
    """
    Checks that parameter is a np.ndarray of datetime

    Parameters
    ----------
    parameter: np.ndarray of datetime
        Input parameter

    Returns
    -------
    is_ok: bool
        True if given parameter is ok

    Raises
    ------
    exception: exception
        If parameter is not valid

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
    if not type(parameter) is np.ndarray:
        raise Exception("`parameter` is not a np.ndarray")

    if parameter.size == 0:
        return True

    if not type(parameter[0]) is datetime:
        raise Exception("`parameter` must contain datetime values")

    return True


def check_same_length_ndarray(arr_1, arr_2):
    """
    Checks that arr_1 and arr_2 have the same length

    Parameters
    ----------
    arr_1: np.ndarray
        First array
    arr_2: np.ndarray
        Second array

    Returns
    -------
    is_ok: bool
        True if given arr_1 and arr_2 are ok

    Raises
    ------
    exception: exception
        If given arr_1 and arr_2 are not valid

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
    if not arr_1.size == arr_2.size:
        raise Exception("Given vectors must have the same length")

    return True


def check_same_length_dataframe(data_1, data_2):
    """
    Checks that data_1 and data_2 have the same length

    Parameters
    ----------
    data_1: pd.Dataframe
        First data frame
    data_2: pd.Dataframe
        Second data frame

    Returns
    -------
    is_ok: bool
        True if given data_1 and data_2 are ok

    Raises
    ------
    exception: exception
        If given data_1 and data_2 are not valid

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
    if not data_1.size == data_2.size:
        raise Exception("Given Dataframes must have the same length")

    return True


def check_comparable_data(data_1, data_2):
    """
    Checks that data_1 and data_2 are comparable

    Parameters
    ----------
    data_1: pd.Dataframe
        First data frame
    data_2: pd.Dataframe
        Second data frame

    Returns
    -------
    is_ok: bool
        True if given data_1 and data_2 are ok

    Raises
    ------
    exception: exception
        If given data_1 and data_2 are not valid

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
    if not data_1.shape[0] == data_2.shape[0]:
        raise Exception("Given dataframes must have the same length")
    if data_1.shape[0] == 0:
        return True
    if not data_1.t.values[0] == data_2.t.values[0]:
        raise Exception("Given dataframes must start from the same timestamp")
    if not data_1.t.values[-1] == data_2.t.values[-1]:
        raise Exception("Given dataframes must end with the same timestamp")

    return True
