import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from copy import copy

from scipy.interpolate import interp1d

from py_agata.inspection import find_nan_islands
from py_agata.input_validator import *

def detrend_glucose(data):
    """
    Detrends glucose data. To do that, the function computes the slope of the immaginary line that "links" the first
    and last glucose datapoints in the timeseries, then it "flatten" the entire timeseries according to that slope.

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe, numpy array or float containing the glucose data

    Returns
    -------
    data: pd.DataFrame
        Pandas dataframe, numpy array or float containing the detrended glucose data

    Raises
    ------
    None

    See Also
    --------
    None
Ï
    Examples
    --------
    None

    References
    ----------
    None
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    # Compute the slope
    first_point = np.where(~np.isnan(data.glucose.values))[0]
    if first_point.size > 1:
        t0 = pd.to_datetime(data.t.values[0]).to_pydatetime()
        t1 = pd.to_datetime(data.t.values[1]).to_pydatetime()
        sample_time = (t1 - t0).total_seconds() / 60

        last_point = first_point[-1]
        first_point = first_point[0]
        m = (data.glucose.values[last_point] - data.glucose.values[first_point]) / ((last_point - first_point) * sample_time)

        # Detrend data
        data_detrented = copy(data)
        data_detrented.glucose[:] = data_detrented.glucose.values - m*np.arange(0,data_detrented.glucose.values.size)*sample_time
        return data_detrented
    else:
        return copy(data)

def impute_glucose(data, max_gap):
    """
    Imputes missing glucose data using linear interpolation. The function imputes only missing data gaps of maximum
    `max_gap` minutes. Gaps longer than `max_gap` minutes are ignored.

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe, numpy array or float containing the glucose data
    max_gap: int
        An integer defining the maximum interpolable missing data gaps (in min)

    Returns
    -------
    data_imputed: pd.DataFrame
        Pandas dataframe, numpy array or float containing the imputed glucose data


    Raises
    ------
    None

    See Also
    --------
    None
Ï
    Examples
    --------
    None

    References
    ----------
    None
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)
    check_int_parameter(max_gap)

    # Get the sample time
    t0 = pd.to_datetime(data.t.values[0]).to_pydatetime()
    t1 = pd.to_datetime(data.t.values[1]).to_pydatetime()
    sample_time = (t1 - t0).total_seconds() / 60

    # Find the interpolable gaps
    short_nan, long_nan, nan_start, nan_end = find_nan_islands(data, int(np.round(max_gap / sample_time)))

    # Impute data
    data_imputed = copy(data)
    idxs = np.arange(data_imputed.shape[0])
    f = interp1d(idxs[~np.isnan(data.glucose.values)], data.glucose.values[~np.isnan(data.glucose.values)], kind='linear')
    data_imputed.glucose.values[short_nan] = f(short_nan)

    return data_imputed

def retime_glucose(data, timestep):
    """
    Retimes the given `data` timetable to a  new timetable with homogeneous `timestep`. It puts nans where glucose
    datapoints are missing and it uses mean to solve conflicts (i.e., when two glucose datapoints have the same retimed timestamp.

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe, numpy array or float containing the glucose data
    timestep: int
        An integer defining the timestep to use in the new timetable.

    Returns
    -------
    data_retimed: pd.DataFrame
        Pandas dataframe, numpy array or float containing the retimed glucose data


    Raises
    ------
    None

    See Also
    --------
    None
Ï
    Examples
    --------
    None

    References
    ----------
    None
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)

    data_temp = copy(data)
    start_time = pd.to_datetime(data_temp.t.values[0]).to_pydatetime()
    start_time = start_time.replace(second=0)
    end_time = pd.to_datetime(data_temp.t.values[-1]).to_pydatetime()

    new_t = np.arange(start_time, end_time, timedelta(minutes=timestep)).astype(datetime)
    values = np.empty(new_t.size)
    values.fill(np.nan)

    dr = {'t': new_t, 'glucose': values, 'k' : values}
    data_retimed = pd.DataFrame(data=dr)

    data_temp = data_temp.drop(np.where(np.isnan(data_temp.glucose.values))[0]).reset_index().drop(columns='index')

    t_list = pd.to_datetime(data_retimed.t.values).to_pydatetime().tolist()

    for t in range(data_temp.shape[0]):

        # Find the nearest timestamp
        t_temp = pd.to_datetime(data_temp.t.values[t]).to_pydatetime()
        distances = [abs(x - t_temp).total_seconds() for x in t_list]
        idx_near = np.where(min(distances) == np.array(distances))[0][0]

        # Manage conflicts computing their average
        if np.isnan(data_retimed.glucose[idx_near]):
            data_retimed.glucose[idx_near] = data_temp.glucose[t]
            data_retimed.k[idx_near] = 1
        else:
            data_retimed.glucose[idx_near] += data_temp.glucose[t]
            data_retimed.k[idx_near] += 1

    # Compute the average and remove column 'k'
    data_retimed.glucose = np.divide(data_retimed.glucose, data_retimed.k)
    data_retimed = data_retimed.drop(columns='k')

    return data_retimed