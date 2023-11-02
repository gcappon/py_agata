import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from copy import copy

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