import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from copy import copy

from py_agata.input_validator import *

def find_nan_islands(data, th):
    """
    Locates nan sequences in vector `data`, and classifies them based on their length (longer or
    not than the specified threshold `th`).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe, numpy array or float containing the glucose data
    th: int
        Threshold, expressed in number of samples, to distinguish between
        long and short nan sequences

    Returns
    -------
    short_nan: np.ndarray
        The indices of "short nan" sequences (i.e., sequences shorter than TH consecutive nan samples)
    long_nan: np.ndarray
        The indices of "long nan" sequences (i.e., sequences longer than or equal to TH consecutive nan samples)
    nan_start: np.ndarray
        Starts of each nan sequence
    nan_end: end of each nan sequence

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
    check_int_parameter(th)

    # Get glucose data
    glucose = data.glucose.values

    # Locate nan sequences
    nan_ind = np.where(np.isnan(glucose))[0]

    if nan_ind.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    elif nan_ind.size == 1:
        if th <= 1:
            return np.array([]), np.array([nan_ind[0]]), np.array([nan_ind[0]]), np.array([nan_ind[0]])
        else:
            return np.array([nan_ind[0]]), np.array([]), np.array([nan_ind[0]]), np.array([nan_ind[0]])
    else:

        short_nan = np.array([])
        long_nan = np.array([])
        nan_start = np.array([])
        nan_end = np.array([])

        start_tmp = nan_ind[0]
        for i in range(1, len(nan_ind)):
            if nan_ind[i] > nan_ind[i-1] + 1:
                nan_start = np.append(nan_start, start_tmp)
                nan_end = np.append(nan_end, nan_ind[i-1])

                if (nan_end[-1] - nan_start[-1] + 1) >= th:
                    long_nan = np.append(long_nan, np.arange(nan_start[-1], nan_end[-1] + 1))
                else:
                    short_nan = np.append(short_nan, np.arange(nan_start[-1], nan_end[-1] + 1))

                start_tmp = nan_ind[i]
        nan_start = np.append(nan_start, start_tmp)
        nan_end = np.append(nan_end, nan_ind[i])

        if (nan_end[-1] - nan_start[-1] + 1) >= th:
            long_nan = np.append(long_nan, np.arange(nan_start[-1], nan_end[-1] + 1))
        else:
            short_nan = np.append(short_nan, np.arange(nan_start[-1], nan_end[-1] + 1))

        return short_nan.astype(int), long_nan.astype(int), nan_start.astype(int), nan_end.astype(int)


def missing_glucose_percentage(data):
    """
    Computes the percentage of missing values in the given glucose trace.

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe, numpy array or float containing the glucose data

    Returns
    -------
    missing_glucose_percentage: float
        The percentage of missing glucose values.

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

    if data.glucose.values.size == 0:
        return np.nan

    return 100 * np.sum(np.isnan(data.glucose))/ data.glucose.values.size


def number_days_of_observation(data):
    """
    Computes the percentage of number of days of observation in the given glucose trace.

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe, numpy array or float containing the glucose data

    Returns
    -------
    number_days_of_observation: float
        The number of days of observation.

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

    if data.glucose.values.size == 0:
        return np.nan

    start_time = pd.to_datetime(data.t.values[0]).to_pydatetime()
    end_time = pd.to_datetime(data.t.values[-1]).to_pydatetime()
    return (end_time - start_time).seconds / (60 * 60 * 24)


def find_hypoglycemic_events(data, th=70.):
    """
    Finds the hypoglycemic events in a given glucose trace. The definition of hypoglycemic event can be found
    in Battellino et al. (event begins: at least consecutive 15 minutes < threshold mg/dl, event ends: at least
    15 consecutive minutes > threshold mg/dl)

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe, numpy array or float containing the glucose data
    th: float, optional, default : 70
        A number defining the threshold to be used to find hypoglycemia (in mg/dl)

    Returns
    -------
    hypoglycemic_events: dict
        A dictionary containing the information on the hypoglycemic events found by the function with fields:
        - time_start: np.ndarray
            The starting timestamps of each found hypoglycemic event
        - time_end: np.ndarray
            The ending timestamps of each found hypoglycemic event
        - duration: np.ndarray
            The duration of each found hypoglycemic event
        - mean_duration: float
            The mean duration of hypoglycemic events
        - number_per_week: float
            The number of hypoglycemic events per week

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
    Battelino et al., "Continuous glucose monitoring and merics for
    clinical trials: An international consensus statement", The Lancet
    Diabetes & Endocrinology, 2022, pp. 1-16.
    DOI: https://doi.org/10.1016/S2213-8587(22)00319-9.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)
    check_float_parameter(th)

    hypoglycemic_events = dict()
    hypoglycemic_events['time_start'] = np.empty(shape=0, dtype=datetime)
    hypoglycemic_events['time_end'] = np.empty(shape=0, dtype=datetime)
    hypoglycemic_events['duration'] = np.empty(shape=0, dtype=float)
    hypoglycemic_events['mean_duration'] = np.nan
    hypoglycemic_events['events_per_week'] = np.nan
    if data.glucose.values.size == 0:
        return hypoglycemic_events

    k = 0 #hypoglycemic_event vector current index
    t0 = pd.to_datetime(data.t.values[0]).to_pydatetime()
    t1 = pd.to_datetime(data.t.values[1]).to_pydatetime()
    sample_time = (t1 - t0).seconds / 60

    n_samples = int(np.round(15/sample_time)) #number of consecutive samples required to define a valid event

    count = 0 #number of current found consecutive samples

    # state flag -> 0: no event, 1: found a valid hypo event and
    #               currently in hypo, -1: found a valid hypo event and
    #               currently not in hypo.
    flag = 0

    for t in range(data.glucose.values.size):

        if data.glucose.values[t] < th:

            # If it is a new event, reset count and set the hypothetical starting time to the current timestamp
            if count <= 0:
                count = 0
                temp_start_time = pd.to_datetime(data.t.values[t]).to_pydatetime()

            count = min([n_samples, count + 1]) #limit count to n_samples

            # if count touches the "nSamples sample goal" we found a new event
            if count == n_samples or flag == -1:
                flag = 1

        else:

            if flag == 0 and count > 0:
                count = 0

            elif flag == 1:
                count = n_samples
                flag = -1

            # countdown
            count = count - 1

        if count == 0 and flag == -1:
            hypoglycemic_events['time_start'] = np.append(hypoglycemic_events['time_start'], temp_start_time)
            hypoglycemic_events['duration'] = np.append(hypoglycemic_events['duration'], (pd.to_datetime(data.t.values[t - (n_samples - 1)]).to_pydatetime() - temp_start_time).seconds / 60)
            k += 1
            flag = 0

    # Manage the case in which the hypoglycemic event has been found (at
    # least n_samples < th) but the post hypoglycemic window has not finshed
    # yet.
    if count > 0 and flag == -1:
        hypoglycemic_events['time_start'] = np.append(hypoglycemic_events['time_start'], temp_start_time)
        hypoglycemic_events['duration'] = np.append(hypoglycemic_events['duration'], (pd.to_datetime(data.t.values[t - (n_samples - count - 1)]).to_pydatetime() - temp_start_time).seconds / 60)
        k += 1

    if count == n_samples and flag == 1:
        hypoglycemic_events['time_start'] = np.append(hypoglycemic_events['time_start'], temp_start_time)
        hypoglycemic_events['duration'] = np.append(hypoglycemic_events['duration'], (pd.to_datetime(data.t.values[t]).to_pydatetime() - temp_start_time).seconds / 60 + sample_time)
        k = k + 1

    for k in range(hypoglycemic_events['time_start'].size):
        hypoglycemic_events['time_end'] = np.append(hypoglycemic_events['time_end'], hypoglycemic_events['time_start'][k] + timedelta(minutes=hypoglycemic_events['duration'][k]))

    if hypoglycemic_events['duration'].size == 0:
        hypoglycemic_events['mean_duration'] = np.nan
    else:
        hypoglycemic_events['mean_duration'] = np.mean(hypoglycemic_events['duration'])
    n_days = number_days_of_observation(data)
    hypoglycemic_events['events_per_week'] = hypoglycemic_events['time_start'].size / n_days * 7

    return hypoglycemic_events


def find_hyperglycemic_events(data, th=180.):
    """
    Finds the hyperglycemic events in a given glucose trace. The definition of hyperglycemic event can be found
    in Battellino et al. (event begins: at least consecutive 15 minutes > threshold mg/dl, event ends: at least
    15 consecutive minutes < threshold mg/dl)

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe, numpy array or float containing the glucose data
    th: float, optional, default : 180
        A number defining the threshold to be used to find hyperglycemia (in mg/dl)

    Returns
    -------
    hyperglycemic_events: dict
        A dictionary containing the information on the hyperglycemic events found by the function with fields:
        - time_start: np.ndarray
            The starting timestamps of each found hyperglycemic event
        - time_end: np.ndarray
            The ending timestamps of each found hyperglycemic event
        - duration: np.ndarray
            The duration of each found hyperglycemic event
        - mean_duration: float
            The mean duration of hyperglycemic events
        - number_per_week: float
            The number of hyperglycemic events per week

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
    Battelino et al., "Continuous glucose monitoring and merics for
    clinical trials: An international consensus statement", The Lancet
    Diabetes & Endocrinology, 2022, pp. 1-16.
    DOI: https://doi.org/10.1016/S2213-8587(22)00319-9.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)
    check_float_parameter(th)

    hyperglycemic_events = dict()
    hyperglycemic_events['time_start'] = np.empty(shape=0, dtype=datetime)
    hyperglycemic_events['time_end'] = np.empty(shape=0, dtype=datetime)
    hyperglycemic_events['duration'] = np.empty(shape=0, dtype=float)
    hyperglycemic_events['mean_duration'] = np.nan
    hyperglycemic_events['events_per_week'] = np.nan
    if data.glucose.values.size == 0:
        return hyperglycemic_events

    k = 0 #hypoglycemic_event vector current index
    t0 = pd.to_datetime(data.t.values[0]).to_pydatetime()
    t1 = pd.to_datetime(data.t.values[1]).to_pydatetime()
    sample_time = (t1 - t0).seconds / 60

    n_samples = int(np.round(15/sample_time)) #number of consecutive samples required to define a valid event

    count = 0 #number of current found consecutive samples

    # state flag -> 0: no event, 1: found a valid hyper event and
    #               currently in hypo, -1: found a valid hyper event and
    #               currently not in hyper.
    flag = 0

    for t in range(data.glucose.values.size):

        if data.glucose.values[t] > th:

            # If it is a new event, reset count and set the hypothetical starting time to the current timestamp
            if count <= 0:
                count = 0
                temp_start_time = pd.to_datetime(data.t.values[t]).to_pydatetime()

            count = min([n_samples, count + 1]) #limit count to n_samples

            # if count touches the "nSamples sample goal" we found a new event
            if count == n_samples or flag == -1:
                flag = 1

        else:

            if flag == 0 and count > 0:
                count = 0

            elif flag == 1:
                count = n_samples
                flag = -1

            # countdown
            count = count - 1

        if count == 0 and flag == -1:
            hyperglycemic_events['time_start'] = np.append(hyperglycemic_events['time_start'], temp_start_time)
            hyperglycemic_events['duration'] = np.append(hyperglycemic_events['duration'], (pd.to_datetime(data.t.values[t - (n_samples - 1)]).to_pydatetime() - temp_start_time).seconds / 60)
            k += 1
            flag = 0

    # Manage the case in which the hyperglycemic event has been found (at
    # least n_samples > th) but the post hyperglycemic window has not finshed
    # yet.
    if count > 0 and flag == -1:
        hyperglycemic_events['time_start'] = np.append(hyperglycemic_events['time_start'], temp_start_time)
        hyperglycemic_events['duration'] = np.append(hyperglycemic_events['duration'], (pd.to_datetime(data.t.values[t - (n_samples - count - 1)]).to_pydatetime() - temp_start_time).seconds / 60)
        k += 1

    if count == n_samples and flag == 1:
        hyperglycemic_events['time_start'] = np.append(hyperglycemic_events['time_start'], temp_start_time)
        hyperglycemic_events['duration'] = np.append(hyperglycemic_events['duration'], (pd.to_datetime(data.t.values[t]).to_pydatetime() - temp_start_time).seconds / 60 + sample_time)
        k = k + 1

    for k in range(hyperglycemic_events['time_start'].size):
        hyperglycemic_events['time_end'] = np.append(hyperglycemic_events['time_end'], hyperglycemic_events['time_start'][k] + timedelta(minutes=hyperglycemic_events['duration'][k]))

    if hyperglycemic_events['duration'].size == 0:
        hyperglycemic_events['mean_duration'] = np.nan
    else:
        hyperglycemic_events['mean_duration'] = np.mean(hyperglycemic_events['duration'])
    n_days = number_days_of_observation(data)
    hyperglycemic_events['events_per_week'] = hyperglycemic_events['time_start'].size / n_days * 7

    return hyperglycemic_events


def find_extended_hypoglycemic_events(data, th=54.):
    """
    Finds the extended hypoglycemic events in a given glucose trace. The definition of hypoglycemic event can be found
    in Battellino et al. (event begins: at least consecutive 120 minutes < threshold mg/dl, event ends: at least
    15 consecutive minutes > threshold mg/dl)

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe, numpy array or float containing the glucose data
    th: float, optional, default : 54
        A number defining the threshold to be used to find hypoglycemia (in mg/dl)

    Returns
    -------
    extended_hypoglycemic_events: dict
        A dictionary containing the information on the extended hypoglycemic events found by the function with fields:
        - time_start: np.ndarray
            The starting timestamps of each found extended hypoglycemic event
        - time_end: np.ndarray
            The ending timestamps of each found extended hypoglycemic event
        - duration: np.ndarray
            The duration of each found extended hypoglycemic event
        - mean_duration: float
            The mean duration of extended hypoglycemic events
        - number_per_week: float
            The number of extended hypoglycemic events per week

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
    Battelino et al., "Continuous glucose monitoring and merics for
    clinical trials: An international consensus statement", The Lancet
    Diabetes & Endocrinology, 2022, pp. 1-16.
    DOI: https://doi.org/10.1016/S2213-8587(22)00319-9.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)
    check_float_parameter(th)

    extended_hypoglycemic_events = dict()
    extended_hypoglycemic_events['time_start'] = np.empty(shape=0, dtype=datetime)
    extended_hypoglycemic_events['time_end'] = np.empty(shape=0, dtype=datetime)
    extended_hypoglycemic_events['duration'] = np.empty(shape=0, dtype=float)
    extended_hypoglycemic_events['mean_duration'] = np.nan
    extended_hypoglycemic_events['events_per_week'] = np.nan
    if data.glucose.values.size == 0:
        return extended_hypoglycemic_events

    k = 0 #hypoglycemic_event vector current index
    t0 = pd.to_datetime(data.t.values[0]).to_pydatetime()
    t1 = pd.to_datetime(data.t.values[1]).to_pydatetime()
    sample_time = (t1 - t0).seconds / 60

    n_samples_in = int(np.round(120/sample_time)) #number of consecutive samples required to define the start of a valid event
    n_samples_out = int(np.round(15/ sample_time))  # number of consecutive samples required to define the end of a valid event

    count = 0 #number of current found consecutive samples

    # state flag -> 0: no event, 1: found a valid hypo event and
    #               currently in hypo, -1: found a valid hypo event and
    #               currently not in hypo.
    flag = 0

    for t in range(data.glucose.values.size):

        if data.glucose.values[t] < th:

            # If it is a new event, reset count and set the hypothetical starting time to the current timestamp
            if count <= 0:
                count = 0
                temp_start_time = pd.to_datetime(data.t.values[t]).to_pydatetime()

            count = min([n_samples_in, count + 1]) #limit count to n_samples

            # if count touches the "nSamples sample goal" we found a new event
            if count == n_samples_in or flag == -1:
                flag = 1

        else:

            if flag == 0 and count > 0:
                count = 0

            elif flag == 1:
                count = n_samples_out
                flag = -1

            # countdown
            count = count - 1

        if count == 0 and flag == -1:
            extended_hypoglycemic_events['time_start'] = np.append(extended_hypoglycemic_events['time_start'], temp_start_time)
            extended_hypoglycemic_events['duration'] = np.append(extended_hypoglycemic_events['duration'], (pd.to_datetime(data.t.values[t - (n_samples_out - 1)]).to_pydatetime() - temp_start_time).seconds / 60)
            k += 1
            flag = 0

    # Manage the case in which the hypoglycemic event has been found (at
    # least n_samples < th) but the post hypoglycemic window has not finshed
    # yet.
    if count > 0 and flag == -1:
        extended_hypoglycemic_events['time_start'] = np.append(extended_hypoglycemic_events['time_start'], temp_start_time)
        extended_hypoglycemic_events['duration'] = np.append(extended_hypoglycemic_events['duration'], (pd.to_datetime(data.t.values[t - (n_samples_out - count - 1)]).to_pydatetime() - temp_start_time).seconds / 60)
        k += 1

    if count == n_samples_in and flag == 1:
        extended_hypoglycemic_events['time_start'] = np.append(extended_hypoglycemic_events['time_start'], temp_start_time)
        extended_hypoglycemic_events['duration'] = np.append(extended_hypoglycemic_events['duration'], (pd.to_datetime(data.t.values[t]).to_pydatetime() - temp_start_time).seconds / 60 + sample_time)
        k = k + 1

    for k in range(extended_hypoglycemic_events['time_start'].size):
        extended_hypoglycemic_events['time_end'] = np.append(extended_hypoglycemic_events['time_end'], extended_hypoglycemic_events['time_start'][k] + timedelta(minutes=extended_hypoglycemic_events['duration'][k]))

    if extended_hypoglycemic_events['duration'].size == 0:
        extended_hypoglycemic_events['mean_duration'] = np.nan
    else:
        extended_hypoglycemic_events['mean_duration'] = np.mean(extended_hypoglycemic_events['duration'])
    n_days = number_days_of_observation(data)
    extended_hypoglycemic_events['events_per_week'] = extended_hypoglycemic_events['time_start'].size / n_days * 7

    return extended_hypoglycemic_events


def find_hypoglycemic_events_by_level(data, glycemic_target = 'diabetes'):
    """
    Finds the hypoglycemic events in a given glucose trace classifying them by level, i.e., hypo, level 1 hypo or level 2 hypo.
    The definition of hypoglycemic event can be found in Battellino et al.

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe, numpy array or float containing the glucose data
    glycemic_target: str, optional, default : 'diabetes'
        A string defining the set of glycemic targets to use.

    Returns
    -------
    hypoglycemic_events: dict
        A dictionary containing the information on the hypoglycemic events found by the function divided by level with fields:
        - hypo: dict
            A dictionary containing the information on the hypoglycemic events found by the function with fields:
            - time_start: np.ndarray
                The starting timestamps of each found hypoglycemic event
            - time_end: np.ndarray
                The ending timestamps of each found hypoglycemic event
            - duration: np.ndarray
                The duration of each found hypoglycemic event
            - mean_duration: float
                The mean duration of hypoglycemic events
            - number_per_week: float
                The number of hypoglycemic events per week
        - l1: dict
            A dictionary containing the information on the l1 hypoglycemic events found by the function with fields:
            - time_start: np.ndarray
                The starting timestamps of each found l1 hypoglycemic event
            - time_end: np.ndarray
                The ending timestamps of each found l1 hypoglycemic event
            - duration: np.ndarray
                The duration of each found l1 hypoglycemic event
            - mean_duration: float
                The mean duration of l1 hypoglycemic events
            - number_per_week: float
                The number of l1 hypoglycemic events per week
        - l2: dict
            A dictionary containing the information on the l2 hypoglycemic events found by the function with fields:
            - time_start: np.ndarray
                The starting timestamps of each found l2 hypoglycemic event
            - time_end: np.ndarray
                The ending timestamps of each found l2 hypoglycemic event
            - duration: np.ndarray
                The duration of each found l2 hypoglycemic event
            - mean_duration: float
                The mean duration of l2 hypoglycemic events
            - number_per_week: float
                The number of l2 hypoglycemic events per week

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
    Battelino et al., "Continuous glucose monitoring and merics for
    clinical trials: An international consensus statement", The Lancet
    Diabetes & Endocrinology, 2022, pp. 1-16.
    DOI: https://doi.org/10.1016/S2213-8587(22)00319-9.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)
    check_str_parameter(glycemic_target)

    if glycemic_target == 'diabetes':
        th_l1 = 70.
        th_l2 = 54.
    else:
        th_l1 = 63.
        th_l2 = 54.

    # Get all hypoglycemic events
    all_hypo_events = find_hypoglycemic_events(data, th=th_l1)

    # Get L2 hypoglycemic events
    l2_hypo_events = find_hypoglycemic_events(data, th=th_l2)

    flag_l1_events = np.full((all_hypo_events['time_start'].size,), True)

    for h in range(l2_hypo_events['time_start'].size):

        distances = [(x - l2_hypo_events['time_start'][h]).total_seconds() for x in all_hypo_events['time_start']]
        idxs = np.where(np.array(distances) < 0)[0]
        if idxs.size > 0:
            flag_l1_events[idxs[-1]] = False

    hypoglycemic_events = dict()
    hypoglycemic_events['hypo'] = copy(all_hypo_events)
    hypoglycemic_events['l1'] = dict()
    hypoglycemic_events['l1']['time_start'] = copy(all_hypo_events['time_start'][flag_l1_events])
    hypoglycemic_events['l1']['time_end'] = copy(all_hypo_events['time_end'][flag_l1_events])
    hypoglycemic_events['l1']['duration'] = copy(all_hypo_events['duration'][flag_l1_events])
    hypoglycemic_events['l1']['mean_duration'] = np.mean(hypoglycemic_events['l1']['duration'])
    n_days = number_days_of_observation(data)
    hypoglycemic_events['l1']['events_per_week'] = hypoglycemic_events['l1']['time_start'].size / n_days * 7
    hypoglycemic_events['l2'] = copy(l2_hypo_events)

    return hypoglycemic_events


def find_hyperglycemic_events_by_level(data, glycemic_target='diabetes'):
    """
    Finds the hyperglycemic events in a given glucose trace classifying them by level, i.e., hyper, level 1 hyper or level 2 hyper.
    The definition of hyperglycemic event can be found in Battellino et al.

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe, numpy array or float containing the glucose data
    glycemic_target: str, optional, default : 'diabetes'
        A string defining the set of glycemic targets to use.

    Returns
    -------
    hyperglycemic_events: dict
        A dictionary containing the information on the hyperglycemic events found by the function divided by level with fields:
        - hyper: dict
            A dictionary containing the information on the hyperglycemic events found by the function with fields:
            - time_start: np.ndarray
                The starting timestamps of each found hyperglycemic event
            - time_end: np.ndarray
                The ending timestamps of each found hyperglycemic event
            - duration: np.ndarray
                The duration of each found hyperglycemic event
            - mean_duration: float
                The mean duration of hyperglycemic events
            - number_per_week: float
                The number of hyperglycemic events per week
        - l1: dict
            A dictionary containing the information on the l1 hyperglycemic events found by the function with fields:
            - time_start: np.ndarray
                The starting timestamps of each found l1 hyperglycemic event
            - time_end: np.ndarray
                The ending timestamps of each found l1 hyperglycemic event
            - duration: np.ndarray
                The duration of each found l1 hyperglycemic event
            - mean_duration: float
                The mean duration of l1 hyperglycemic events
            - number_per_week: float
                The number of l1 hyperglycemic events per week
        - l2: dict
            A dictionary containing the information on the l2 hyperglycemic events found by the function with fields:
            - time_start: np.ndarray
                The starting timestamps of each found l2 hyperglycemic event
            - time_end: np.ndarray
                The ending timestamps of each found l2 hyperglycemic event
            - duration: np.ndarray
                The duration of each found l2 hyperglycemic event
            - mean_duration: float
                The mean duration of l2 hyperglycemic events
            - number_per_week: float
                The number of l2 hyperglycemic events per week

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
    Battelino et al., "Continuous glucose monitoring and merics for
    clinical trials: An international consensus statement", The Lancet
    Diabetes & Endocrinology, 2022, pp. 1-16.
    DOI: https://doi.org/10.1016/S2213-8587(22)00319-9.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)
    check_str_parameter(glycemic_target)

    if glycemic_target == 'diabetes':
        th_l1 = 180.
        th_l2 = 250.
    else:
        th_l1 = 140.
        th_l2 = 250.

    # Get all hyperglycemic events
    all_hyper_events = find_hyperglycemic_events(data, th=th_l1)

    # Get L2 hyperglycemic events
    l2_hyper_events = find_hyperglycemic_events(data, th=th_l2)

    flag_l1_events = np.full((all_hyper_events['time_start'].size,), True)

    for h in range(l2_hyper_events['time_start'].size):

        distances = [(x - l2_hyper_events['time_start'][h]).total_seconds() for x in all_hyper_events['time_start']]
        idxs = np.where(np.array(distances) < 0)[0]
        if idxs.size > 0:
            flag_l1_events[idxs[-1]] = False

    hyperglycemic_events = dict()
    hyperglycemic_events['hyper'] = copy(all_hyper_events)
    hyperglycemic_events['l1'] = dict()
    hyperglycemic_events['l1']['time_start'] = copy(all_hyper_events['time_start'][flag_l1_events])
    hyperglycemic_events['l1']['time_end'] = copy(all_hyper_events['time_end'][flag_l1_events])
    hyperglycemic_events['l1']['duration'] = copy(all_hyper_events['duration'][flag_l1_events])
    hyperglycemic_events['l1']['mean_duration'] = np.mean(hyperglycemic_events['l1']['duration'])
    n_days = number_days_of_observation(data)
    hyperglycemic_events['l1']['events_per_week'] = hyperglycemic_events['l1']['time_start'].size / n_days * 7
    hyperglycemic_events['l2'] = copy(l2_hyper_events)

    return hyperglycemic_events
