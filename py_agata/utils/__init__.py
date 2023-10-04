import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from copy import copy

def to_mmol_l(data):
    """
    Converts a pandas dataframe, numpy array or float containing the glucose data in mgl/dl to mmol/l

    Parameters
    ----------
    data: pd.DataFrame, np.ndarray, float
        Pandas dataframe, numpy array or float containing the glucose data in mg/dl

    Returns
    -------
    data: pd.DataFrame, np.ndarray, float
        Pandas dataframe, numpy array or float containing the glucose data in mmol/l

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
    if type(data) is pd.DataFrame:
        return pd.DataFrame(data={'t': data.t.values, 'glucose': data.glucose.values/18.018})
    else:
        return data/18.018


def to_mg_dl(data):
    """
    Converts a pandas dataframe, numpy array or float containing the glucose data in mmol/l to mg/dl

    Parameters
    ----------
    data: pd.DataFrame, np.ndarray, float
        Pandas dataframe, numpy array or float containing the glucose data in mmol/l

    Returns
    -------
    data: pd.DataFrame, np.ndarray, float
        Pandas dataframe, numpy array or float containing the glucose data in mg/dl

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
    if type(data) is pd.DataFrame:
        return pd.DataFrame(data={'t': data.t.values, 'glucose': data.glucose.values*18.018})
    else:
        return data*18.018


def glucose_time_vectors_to_dataframe(glucose, t):
    """
    Converts the two given vectors containing the glucose samples and the corresponding timestamps, respectively,
    in a dataframe.

    Parameters
    ----------
    glucose: np.ndarray
        A vector of double containing the glucose data (in mg/dl)
    t: np.ndarray
        A vector of datetime containing the timestamps

    Returns
    -------
    data: pd.DataFrame
        Pandas dataframe containing the glucose data in mg/dl

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
    return pd.DataFrame(data={'t': t, 'glucose': glucose})


def glucose_vector_to_dataframe(glucose, sample_time, start_time=datetime(2000, 1, 1, 0, 0, 0)):
    """
    Transform a vector containing glucose samples sampled on an homogeneous timegrid with, timestep
    `sample_time`, in a timetable. The resulting timetable timestamps will
    start from `start_time`. If start_time is not specified, 2000-01-01 00:00 is
    used as default.

    Parameters
    ----------
    glucose: np.ndarray
        A vector of double containing the glucose data (in mg/dl)
    sample_time: int
        The sample time of data (in min)
    start_time: datetime, optional, default: datetime(2000, 1, 1, 0, 0, 0)
        The first timestamp of the resulting timetable. If start_time is not provided, 2000-01-01 00:00, is used as default.

    Returns
    -------
    data: pd.DataFrame
        Pandas dataframe containing the glucose data in mg/dl

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
    time_interval = timedelta(minutes=sample_time)
    end_time = start_time + timedelta(minutes=(glucose.size*sample_time - sample_time))
    t = pd.date_range(start_time, end_time, freq=time_interval)

    return pd.DataFrame(data={'t': t, 'glucose': glucose})


def read_dexcom_data(file, extension='xlsx'):
    """
    Reads data from a .xlsx or .csv file downloaded from the Dexcom CGM system and converts it in a timetable compatible
    with AGATA.

    Parameters
    ----------
    file: str
        A string containing the relative path to the file to be converted in a timetable compatible with AGATA.
    extension: str, optional, {'xlsx', 'csv'}, default: xlsx
        The extension of the raw file.

    Returns
    -------
    data: pd.DataFrame
        Pandas dataframe containing the glucose data in mg/dl

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
    df = pd.read_excel(file) if extension == 'xlsx' else pd.read_csv(file)

    egvs = np.where(df[df.columns[2]] == 'EGV')[0]
    g_raw = df[df.columns[7]][egvs]
    t_raw = df[df.columns[1]][egvs]


    t = np.empty(shape=(t_raw.size,), dtype=datetime)
    glucose = np.empty(shape=(g_raw.size,), dtype=float)

    count = 0
    for e in egvs:

        t[count] = datetime.strptime(t_raw[e], '%Y-%m-%dT%H:%M:%S')

        if type(g_raw[e]) is str:
            if g_raw[e] == 'Low':
                d = 39
            elif g_raw[e] == 'High':
                d = 401
        else:
            d = g_raw[e]
        glucose[count] = d
        count += 1

    t = t[0:count]
    glucose = glucose[0:count]

    data = pd.DataFrame(data={'t': t, 'glucose': glucose})
    data = data.sort_values(by='t')
    return data


def read_eversense_data(file, extension='xlsx'):
    """
    Reads data from a .xlsx or .csv file downloaded from the Eversense CGM system and converts it in a timetable compatible
    with AGATA.

    Parameters
    ----------
    file: str
        A string containing the relative path to the file to be converted in a timetable compatible with AGATA.
    extension: str, optional, {'xlsx', 'csv'}, default: xlsx
        The extension of the raw file.

    Returns
    -------
    data: pd.DataFrame
        Pandas dataframe containing the glucose data in mg/dl

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
    df = pd.read_excel(file) if extension == 'xlsx' else pd.read_csv(file)

    g_raw = df[df.columns[2]]
    t_date_raw = df[df.columns[0]]
    t_time_raw = df[df.columns[1]]
    unit_raw = df[df.columns[3]]

    t = np.empty(shape=(g_raw.size,), dtype=datetime)
    glucose = np.empty(shape=(g_raw.size,), dtype=float)

    count = 0
    for i in range(0, g_raw.size):

        glucose[count] = g_raw[i] if unit_raw[i] == 'mg/dL' else to_mg_dl(g_raw[i])

        t[count] = datetime.strptime(t_date_raw[i] + ' ' + t_time_raw[i], '%d-%B-%Y %I:%M %p')
        count += 1

    t = t[0:count]
    glucose = glucose[0:count]

    data = pd.DataFrame(data={'t': t, 'glucose': glucose})
    data = data.sort_values(by='t')
    return data


def read_freestyle_libre_data(file, extension='xlsx'):
    """
    Reads data from a .xlsx or .csv file downloaded from the Freestyle Libre CGM system and converts it in a timetable compatible
    with AGATA.

    Parameters
    ----------
    file: str
        A string containing the relative path to the file to be converted in a timetable compatible with AGATA.
    extension: str, optional, {'xlsx', 'csv'}, default: xlsx
        The extension of the raw file.

    Returns
    -------
    data: pd.DataFrame
        Pandas dataframe containing the glucose data in mg/dl

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
    df = pd.read_excel(file) if extension == 'xlsx' else pd.read_csv(file)

    g_raw = df[df.columns[6]][2:]
    t_date_raw = df[df.columns[3]][2:].to_dict()
    t_time_raw = df[df.columns[4]][2:].to_dict()

    t = np.empty(shape=(g_raw.size,), dtype=datetime)
    glucose = np.empty(shape=(g_raw.size,), dtype=float)

    count = 0
    for key in t_date_raw:

        glucose[count] = g_raw.values[count]
        t[count] = t_date_raw[key].to_pydatetime()+timedelta(minutes=t_time_raw[key].minute, hours=t_time_raw[key].hour)
        count += 1

    t = t[0:count]
    glucose = glucose[0:count]

    data = pd.DataFrame(data={'t': t, 'glucose': glucose})
    data = data.sort_values(by='t')
    return data