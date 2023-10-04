import numpy as np
import pandas as pd
from scipy.stats import iqr
from scipy.signal import find_peaks
from datetime import timedelta

from py_agata.time_in_ranges import time_in_target, time_in_hypoglycemia


def mean_glucose(data):
    """
    Computes the mean glucose level (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    mean_glucose: float
        The mean glucose level.

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
    Wikipedia on mean: https://en.wikipedia.org/wiki/Mean (Accessed: 2020-12-10).
    """
    # Get non-nan values
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Return nan if all values are nan
    if values.size == 0:
        return np.nan

    # Return the result
    return np.mean(values)


def median_glucose(data):
    """
    Computes the median glucose level (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    mean_glucose: float
        The median glucose level.

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
    Wikipedia on median: https://en.wikipedia.org/wiki/Median (Accessed: 2020-12-10).
    """
    # Get non-nan values
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Return nan if all values are nan
    if values.size == 0:
        return np.nan

    # Return the result
    return np.median(values)


def std_glucose(data):
    """
    Computes the std glucose level (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    std_glucose: float
        The std glucose level.

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
    Wikipedia on standard deviation: https://en.wikipedia.org/wiki/Standard_deviation (Accessed: 2020-12-10).
    """
    # Get non-nan values
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Return nan if all values are nan
    if values.size == 0:
        return np.nan

    # Return the result
    return np.std(values, ddof=1)


def cv_glucose(data):
    """
    Computes the coefficient of variation of glucose (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    cv_glucose: float
        The cv of glucose.

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
    Wikipedia on coefficient of variation: https://en.wikipedia.org/wiki/Coefficient_of_variation (Accessed: 2020-12-10).
    """
    # Return the result
    return 100 * std_glucose(data) / mean_glucose(data)


def range_glucose(data):
    """
    Computes the spanned glucose range (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    range: float
        The range of glucose.

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
    Wikipedia on range: https://en.wikipedia.org/wiki/Range_(statistics) (Accessed: 2020-12-10).
    """
    # Return the result
    return np.nanmax(data.glucose.values) - np.nanmin(data.glucose.values)


def iqr_glucose(data):
    """
    Computes the interquartile range of glucose values (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    iqr_glucose: float
        The interquartile range of glucose.

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
    Wikipedia on IQR: https://en.wikipedia.org/wiki/Interquartile_range (Accessed: 2020-12-10).
    """

    # Get rid of nans
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Return the result
    return iqr(values)


def auc_glucose_over_basal(data, basal):
    """
    Computes the area under the glucose curve using a given basal offset (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    auc_glucose_over_basal: float
        The area under the glucose curve.

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
    # Get non-nan values
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Return nan if all values are nan
    if values.size == 0:
        return np.nan

    # Shift the trace
    values = values - basal

    # Get ts
    ts = (data.t.to_dict()[1].to_pydatetime() - data.t.to_dict()[0].to_pydatetime()).total_seconds() / 60

    # Return the result
    return np.sum(values*ts)


def auc_glucose(data):
    """
    Computes the area under the glucose curve (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    auc_glucose: float
        The area under the glucose curve.

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
    # Return results
    return auc_glucose_over_basal(data, 0)


def gmi(data):
    """
    Computes the glucose management indicator of the given data (ignoring nan values).
    It should be computed only if more than 12 days are available.

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    gmi: float
        The glucose management indicator of the given data.

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
    Bergenstal et al., "Glucose Management Indicator (GMI): A new term
    for estimating A1C from continuous glucose monitoring", Diabetes Care,
    2018, vol. 41, pp. 2275-2280. DOI: 10.2337/dc18-1581.
    """
    # Return results
    return 3.31 + 0.02392 * mean_glucose(data)


def cogi(data):
    """
    Computes the Continuous Glucose Monitoring Index (COGI) of the given data (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    cogi: float
        The Continuous Glucose Monitoring Index (COGI) of the given data.

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
    Leelaranthna et al., "Evaluating glucose control with a novel composite
    Continuous Glucose Monitoring Index", Journal of Diabetes Science and Technology,
    2019, vol. 14, pp. 277-283. DOI: 10.1177/1932296819838525.
    """
    # Compute TIR component
    tir = time_in_target(data)*0.5

    # Compute TBR component
    tbr = np.min([15, time_in_hypoglycemia(data)])
    tbr = (100 - 100 / 15 * tbr) * 0.35

    # Compute GV component
    gv = np.min([np.max([std_glucose(data) / 18.018, 1]), 6])
    gv = (120 - 20 * gv) * 0.15

    # Return results
    return tir + tbr + gv


def conga(data):
    """
    Computes the Continuous Overall Net Glycemic Action (CONGA) of the given data (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    conga: float
        The Continuous Overall Net Glycemic Action (CONGA) of the given data.

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
    McDonnell et al., "A novel approach to continuous glucose analysis
    utilizing glycemic variation", Diabetes Technol Ther, 2005, vol. 7,
    pp. 253–263. DOI: 10.1089/dia.2005.7.253.
    """
    # Set the CONGAOrd hyperparameter to 4 (number of hours in the past it
    # refers to)
    conga_ord = 4

    # Build vectors
    n = data.glucose.values.size
    dc = np.empty(shape=(0,))

    for i in range(1,n):

        # Find the index referring to conga_ord hours ago
        j = np.where(data.t <= (data.t.to_dict()[i].to_pydatetime() - timedelta(hours=conga_ord)))[0]

        if not j.size == 0:
            j = j[-1]
            dc = np.append(dc, data.glucose.values[i] - data.glucose[j])

    # Return results
    if dc.size == 0:
        return np.nan
    else:
        return np.nanstd(dc, ddof=1)


def j_index(data):
    """
    Computes the J-Index of the given data (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    j_index: float
        The J-Index of the given data.

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
    - Wójcicki, "J-index. A new proposition of the assessment of current
    glucose control in diabetic patients", Hormone and Metabolic Reseach,
    1995, vol. 27, pp. 41-42. DOI: 10.1055/s-2007-979906.
    """
    return 1e-3 * (mean_glucose(data) + std_glucose(data)) ** 2


def mage_plus_index(data):
    """
    Computes the mean amplitude of positive glycemic excursion (MAGE+) index of the given data (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    mage_plus_index: float
        The mean amplitude of positive glycemic excursion (MAGE+) index of the given data.

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
    - Service et al., "Mean amplitude of glycemic excursions, a measure of
    diabetic instability", Diabetes, 1970, vol. 19, pp. 644-655. DOI:
    10.2337/diab.19.9.644.
    """

    # Get the first and last day limits
    first_day = data.t.to_dict()[0].to_pydatetime()
    first_day = first_day.replace(hour=0, minute=0, second=0)
    last_day = data.t.to_dict()[data.shape[0] - 1].to_pydatetime()
    last_day = first_day.replace(day=last_day.day + 1, hour=0, minute=0, second=0)

    # Calculate the number of days and preallocate
    n_days = (last_day - first_day).days
    mage_day_plus = np.empty(shape=(n_days,))

    for d in range(0, n_days):

        # Step 0: parameters

        # Get the day of data
        low_limit = data.t >= (first_day + timedelta(days=d))
        high_limit = data.t < (first_day + timedelta(days=d + 1))
        flags = np.logical_and(low_limit, high_limit)
        day_data = data.glucose.values[flags]

        # Get glucose values (might be nan)
        std_within = np.nanstd(day_data, ddof=1)
        n = day_data.size

        if n > 3:
            # Step 1: turning points are only local extrema
            i_max = find_peaks(day_data)[0]
            i_min = find_peaks(-day_data)[0]
            i_turning = np.union1d([0, n-1], np.union1d(i_max, i_min)).astype(int)

            turning = day_data[i_turning]
            n_turning = i_turning.size

            # Step 2: Turning points of no interest are removed
            # A turning point is removed if it's not significantly different from
            # BOTH its left and right-hand side RETAINED neighbours.

            to_be_kept = [True] * n_turning

            for i in range(1, n_turning-1): # First and last samples are retained

                condition_1 = abs(turning[i] - turning[i - 1]) < std_within
                condition_2 = abs(turning[i + 1] - turning[i]) < std_within
                to_be_kept[i] = not(condition_1 and condition_2)

            i_turning = i_turning[to_be_kept]

            # Step 3: Turning points are removed again or moved appropriately
            i = 1
            while i < len(i_turning)-1:
                prev = i_turning[i - 1]
                curr = i_turning[i]
                next = i_turning[i + 1]
                prev_slope = day_data[curr] - day_data[prev]
                next_slope = day_data[next] - day_data[curr]

                if prev_slope < 0 and next_slope > 0:  # Minimum
                    # The actual current turning point is the min in the interval
                    temp = np.nanargmin(day_data[prev:(next+1)]) + prev
                    curr = temp
                    i_turning[i] = curr
                    # The actual previous turning point is the max to the left of the current turning point.
                    temp = np.nanargmax(day_data[prev:curr]) + prev
                    i_turning[i - 1] = temp
                    # The actual following turning point is the max to the right of the current turning point.
                    temp = np.nanargmax(day_data[(curr + 1):(next+1)]) + curr + 1
                    i_turning[i + 1] = temp

                    i += 1
                elif prev_slope > 0 and next_slope < 0:  # Maximum
                    # The actual current turning point is the max in the interval
                    temp = np.nanargmax(day_data[prev:(next+1)]) + prev
                    curr = temp
                    i_turning[i] = curr
                    # The actual previous turning point is the min to the left of the current turning point.
                    temp = np.nanargmin(day_data[prev:curr]) + prev
                    i_turning[i - 1] = temp
                    # The actual following turning point is the min to the right of the current turning point.
                    temp = np.nanargmin(day_data[(curr + 1):(next+1)]) + curr + 1
                    i_turning[i + 1] = temp

                    i += 1
                else:  # Middle point
                    i_turning = np.delete(i_turning, i)

            # Step 4: Remove residual spurious turning points.
            # Turning points not significantly different from EITHER neighbour are
            # removed. Some extra processing is needed for the first and last sample.
            sample1 = day_data[i_turning[0]]
            sample2 = day_data[i_turning[1]]

            if abs(sample2 - sample1) < std_within:
                i_turning = np.delete(i_turning, 0)

            if len(i_turning) > 1:
                # Last sample processing
                sample1 = day_data[i_turning[-2]]
                sample2 = day_data[i_turning[-1]]
                if abs(sample2 - sample1) < std_within:
                    i_turning = np.delete(i_turning, len(i_turning)-1)

            turning = day_data[i_turning]
            n_turning = len(i_turning)

            # Internal points
            to_be_kept = np.ones(n_turning, dtype=bool)
            for i in range(1, n_turning - 1):
                condition1 = abs(turning[i] - turning[i - 1]) < std_within
                condition2 = abs(turning[i + 1] - turning[i]) < std_within
                to_be_kept[i] = not(condition1 or condition2)

            i_turning = i_turning[to_be_kept]
            turning = day_data[i_turning]

            # Step 5: Compute daily MAGE+
            excursions = np.diff(turning)
            mage_day_plus[d] = np.nanmean(excursions[excursions > 0])
        else:
            mage_day_plus[d] = np.nan

    # Compute index
    mage_day_plus[np.isnan(mage_day_plus)] = 0  # Correct for 'mean' behavior
    mpi = np.mean(mage_day_plus)

    # Manage all nan data
    if np.all(np.isnan(data['glucose'])):
        mpi = np.nan

    return mpi


def mage_minus_index(data):
    """
    Computes the mean amplitude of negative glycemic excursion (MAGE-) index of the given data (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    mage_minus_index: float
        The mean amplitude of negative glycemic excursion (MAGE+) index of the given data.

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
    - Service et al., "Mean amplitude of glycemic excursions, a measure of
    diabetic instability", Diabetes, 1970, vol. 19, pp. 644-655. DOI:
    10.2337/diab.19.9.644.
    """

    # Get the first and last day limits
    first_day = data.t.to_dict()[0].to_pydatetime()
    first_day = first_day.replace(hour=0, minute=0, second=0)
    last_day = data.t.to_dict()[data.shape[0] - 1].to_pydatetime()
    last_day = first_day.replace(day=last_day.day + 1, hour=0, minute=0, second=0)

    # Calculate the number of days and preallocate
    n_days = (last_day - first_day).days
    mage_day_minus = np.empty(shape=(n_days,))

    for d in range(0, n_days):

        # Step 0: parameters

        # Get the day of data
        low_limit = data.t >= (first_day + timedelta(days=d))
        high_limit = data.t < (first_day + timedelta(days=d + 1))
        flags = np.logical_and(low_limit, high_limit)
        day_data = data.glucose.values[flags]

        # Get glucose values (might be nan)
        std_within = np.nanstd(day_data, ddof=1)
        n = day_data.size

        if n > 3:
            # Step 1: turning points are only local extrema
            i_max = find_peaks(day_data)[0]
            i_min = find_peaks(-day_data)[0]
            i_turning = np.union1d([0, n-1], np.union1d(i_max, i_min)).astype(int)

            turning = day_data[i_turning]
            n_turning = i_turning.size

            # Step 2: Turning points of no interest are removed
            # A turning point is removed if it's not significantly different from
            # BOTH its left and right-hand side RETAINED neighbours.

            to_be_kept = [True] * n_turning

            for i in range(1, n_turning-1): # First and last samples are retained

                condition_1 = abs(turning[i] - turning[i - 1]) < std_within
                condition_2 = abs(turning[i + 1] - turning[i]) < std_within
                to_be_kept[i] = not(condition_1 and condition_2)

            i_turning = i_turning[to_be_kept]

            # Step 3: Turning points are removed again or moved appropriately
            i = 1
            while i < len(i_turning)-1:
                prev = i_turning[i - 1]
                curr = i_turning[i]
                next = i_turning[i + 1]
                prev_slope = day_data[curr] - day_data[prev]
                next_slope = day_data[next] - day_data[curr]

                if prev_slope < 0 and next_slope > 0:  # Minimum
                    # The actual current turning point is the min in the interval
                    temp = np.nanargmin(day_data[prev:(next+1)]) + prev
                    curr = temp
                    i_turning[i] = curr
                    # The actual previous turning point is the max to the left of the current turning point.
                    temp = np.nanargmax(day_data[prev:curr]) + prev
                    i_turning[i - 1] = temp
                    # The actual following turning point is the max to the right of the current turning point.
                    temp = np.nanargmax(day_data[(curr + 1):(next+1)]) + curr + 1
                    i_turning[i + 1] = temp

                    i += 1
                elif prev_slope > 0 and next_slope < 0:  # Maximum
                    # The actual current turning point is the max in the interval
                    temp = np.nanargmax(day_data[prev:(next+1)]) + prev
                    curr = temp
                    i_turning[i] = curr
                    # The actual previous turning point is the min to the left of the current turning point.
                    temp = np.nanargmin(day_data[prev:curr]) + prev
                    i_turning[i - 1] = temp
                    # The actual following turning point is the min to the right of the current turning point.
                    temp = np.nanargmin(day_data[(curr + 1):(next+1)]) + curr + 1
                    i_turning[i + 1] = temp

                    i += 1
                else:  # Middle point
                    i_turning = np.delete(i_turning, i)

            # Step 4: Remove residual spurious turning points.
            # Turning points not significantly different from EITHER neighbour are
            # removed. Some extra processing is needed for the first and last sample.
            sample1 = day_data[i_turning[0]]
            sample2 = day_data[i_turning[1]]

            if abs(sample2 - sample1) < std_within:
                i_turning = np.delete(i_turning, 0)

            if len(i_turning) > 1:
                # Last sample processing
                sample1 = day_data[i_turning[-2]]
                sample2 = day_data[i_turning[-1]]
                if abs(sample2 - sample1) < std_within:
                    i_turning = np.delete(i_turning, len(i_turning)-1)

            turning = day_data[i_turning]
            n_turning = len(i_turning)

            # Internal points
            to_be_kept = np.ones(n_turning, dtype=bool)
            for i in range(1, n_turning - 1):
                condition1 = abs(turning[i] - turning[i - 1]) < std_within
                condition2 = abs(turning[i + 1] - turning[i]) < std_within
                to_be_kept[i] = not(condition1 or condition2)

            i_turning = i_turning[to_be_kept]
            turning = day_data[i_turning]

            # Step 5: Compute daily MAGE+
            excursions = np.diff(turning)
            mage_day_minus[d] = np.nanmean(excursions[excursions < 0])
        else:
            mage_day_minus[d] = np.nan

    # Compute index
    mage_day_minus[np.isnan(mage_day_minus)] = 0  # Correct for 'mean' behavior
    mmi = -np.mean(mage_day_minus)

    # Manage all nan data
    if np.all(np.isnan(data['glucose'])):
        mmi = np.nan

    return mmi


def mage_index(data):
    """
    Computes the mean amplitude of glycemic excursion (MAGE) index of the given data (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    mage_index: float
        The mean amplitude of glycemic excursion (MAGE) index of the given data.

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
    - Service et al., "Mean amplitude of glycemic excursions, a measure of
    diabetic instability", Diabetes, 1970, vol. 19, pp. 644-655. DOI:
    10.2337/diab.19.9.644.
    """
    return np.nanmean([mage_plus_index(data), mage_minus_index(data)])


def ef_index(data):
    """
    Computes the excursion frequency (EF) index of the given data, i.e., the number of excursion > 75 (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    ef_index: float
        The excursion frequency (EF) index of the given data.

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
    - Service et al., "Mean amplitude of glycemic excursions, a measure of
    diabetic instability", Diabetes, 1970, vol. 19, pp. 644-655. DOI:
    10.2337/diab.19.9.644.
    """
    # Set the fixed parameter
    ef_th = 75

    # Get the first and last day limits
    first_day = data.t.to_dict()[0].to_pydatetime()
    first_day = first_day.replace(hour=0, minute=0, second=0)
    last_day = data.t.to_dict()[data.shape[0] - 1].to_pydatetime()
    last_day = first_day.replace(day=last_day.day + 1, hour=0, minute=0, second=0)

    # Calculate the number of days and preallocate
    n_days = (last_day - first_day).days
    ef_day = np.empty(shape=(n_days,))

    for d in range(0, n_days):

        # Step 0: parameters

        # Get the day of data
        low_limit = data.t >= (first_day + timedelta(days=d))
        high_limit = data.t < (first_day + timedelta(days=d + 1))
        flags = np.logical_and(low_limit, high_limit)
        day_data = data.glucose.values[flags]

        # Get glucose values (might be nan)
        std_within = np.nanstd(day_data, ddof=1)
        n = day_data.size

        if n > 3:
            # Step 1: turning points are only local extrema
            i_max = find_peaks(day_data)[0]
            i_min = find_peaks(-day_data)[0]
            i_turning = np.union1d([0, n-1], np.union1d(i_max, i_min)).astype(int)

            turning = day_data[i_turning]
            n_turning = i_turning.size

            # Step 2: Turning points of no interest are removed
            # A turning point is removed if it's not significantly different from
            # BOTH its left and right-hand side RETAINED neighbours.

            to_be_kept = [True] * n_turning

            for i in range(1, n_turning-1): # First and last samples are retained

                condition_1 = abs(turning[i] - turning[i - 1]) < std_within
                condition_2 = abs(turning[i + 1] - turning[i]) < std_within
                to_be_kept[i] = not(condition_1 and condition_2)

            i_turning = i_turning[to_be_kept]

            # Step 3: Turning points are removed again or moved appropriately
            i = 1
            while i < len(i_turning)-1:
                prev = i_turning[i - 1]
                curr = i_turning[i]
                next = i_turning[i + 1]
                prev_slope = day_data[curr] - day_data[prev]
                next_slope = day_data[next] - day_data[curr]

                if prev_slope < 0 and next_slope > 0:  # Minimum
                    # The actual current turning point is the min in the interval
                    temp = np.nanargmin(day_data[prev:(next+1)]) + prev
                    curr = temp
                    i_turning[i] = curr
                    # The actual previous turning point is the max to the left of the current turning point.
                    temp = np.nanargmax(day_data[prev:curr]) + prev
                    i_turning[i - 1] = temp
                    # The actual following turning point is the max to the right of the current turning point.
                    temp = np.nanargmax(day_data[(curr + 1):(next+1)]) + curr + 1
                    i_turning[i + 1] = temp

                    i += 1
                elif prev_slope > 0 and next_slope < 0:  # Maximum
                    # The actual current turning point is the max in the interval
                    temp = np.nanargmax(day_data[prev:(next+1)]) + prev
                    curr = temp
                    i_turning[i] = curr
                    # The actual previous turning point is the min to the left of the current turning point.
                    temp = np.nanargmin(day_data[prev:curr]) + prev
                    i_turning[i - 1] = temp
                    # The actual following turning point is the min to the right of the current turning point.
                    temp = np.nanargmin(day_data[(curr + 1):(next+1)]) + curr + 1
                    i_turning[i + 1] = temp

                    i += 1
                else:  # Middle point
                    i_turning = np.delete(i_turning, i)

            # Step 4: Remove residual spurious turning points.
            # Turning points not significantly different from EITHER neighbour are
            # removed. Some extra processing is needed for the first and last sample.
            sample1 = day_data[i_turning[0]]
            sample2 = day_data[i_turning[1]]

            if abs(sample2 - sample1) < std_within:
                i_turning = np.delete(i_turning, 0)

            if len(i_turning) > 1:
                # Last sample processing
                sample1 = day_data[i_turning[-2]]
                sample2 = day_data[i_turning[-1]]
                if abs(sample2 - sample1) < std_within:
                    i_turning = np.delete(i_turning, len(i_turning)-1)

            turning = day_data[i_turning]
            n_turning = len(i_turning)

            # Internal points
            to_be_kept = np.ones(n_turning, dtype=bool)
            for i in range(1, n_turning - 1):
                condition1 = abs(turning[i] - turning[i - 1]) < std_within
                condition2 = abs(turning[i + 1] - turning[i]) < std_within
                to_be_kept[i] = not(condition1 or condition2)

            i_turning = i_turning[to_be_kept]
            turning = day_data[i_turning]

            # Step 5: Compute daily MAGE+
            excursions = np.diff(turning)
            ef_day[d] = np.where(abs(excursions) > ef_th)[0].size
        else:
            ef_day[d] = np.nan

    # Compute index
    ei = np.nansum(ef_day)/n_days

    # Manage all nan data
    if np.all(np.isnan(data['glucose'])):
        ei = np.nan

    return ei


def modd(data):
    """
    Computes the mean of daily differences (MODD) of the given data (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    modd: float
        The mean of daily differences (MODD) of the given data.

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
    - Molnar et al., "Day-to-day variation of continuously monitored
    glycaemia: a further measure of diabetic instability", Diabetologia,
    1972, vol. 8, pp. 342–348. DOI: 10.1007/BF01218495.
    """
    # Build vectors
    yesterday = timedelta(minutes=1440)

    n = len(data)

    Dm = []


    for i in range(1, n):

        # Find the index referring to the same time yesterday
        data.t.to_dict()[0].to_pydatetime()
        j = np.where(data.t <= (data.t.to_dict()[i].to_pydatetime() - yesterday))[0]

        if j.size > 0:  # if there is a meaningful sample in data[j]
            j = j[-1]
            Dm.append(abs(data.glucose.values[i] - data.glucose.values[j]))

    if Dm:
        modd = np.nanmean(Dm)
    else:
        modd = np.nan
    return modd


def sddm_index(data):
    """
    Computes the standard deviation of within-day means (SDDM) index of the given data (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    sddm_index: float
        The standard deviation of within-day means (SDDM) index of the given data.

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
    - Rodbard et al., "New and improved methods to characterize glycemic
    variability using continuous glucose monitoring", Diabetes Technology &
    Therapeutics, 2009, vol. 11, pp. 551-565. DOI: 10.1089/dia.2009.0015.
    """
    # Get the first and last day limits
    first_day = data.t.to_dict()[0].to_pydatetime()
    first_day = first_day.replace(hour=0, minute=0, second=0)
    last_day = data.t.to_dict()[data.shape[0] - 1].to_pydatetime()
    last_day = first_day.replace(day=last_day.day + 1, hour=0, minute=0, second=0)

    # Calculate the number of days and preallocate
    n_days = (last_day - first_day).days
    mean_within = np.zeros(shape=(n_days,))

    for d in range(0, n_days):

        # Get the day of data
        low_limit = data.t >= (first_day + timedelta(days=d))
        high_limit = data.t < (first_day + timedelta(days=d + 1))
        flags = np.logical_and(low_limit, high_limit)
        day_data = data.glucose.values[flags]

        # Get daily mean and std
        mean_within[d] = np.nanmean(day_data)

    # Compute index
    return np.nanstd(mean_within, ddof=1)


def sdw_index(data):
    """
    Computes the mean of within-day standard deviation (SDW) index of the given data (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    sdw_index: float
        The mean of within-day standard deviation (SDW) index of the given data.

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
    - Rodbard et al., "New and improved methods to characterize glycemic
    variability using continuous glucose monitoring", Diabetes Technology &
    Therapeutics, 2009, vol. 11, pp. 551-565. DOI: 10.1089/dia.2009.0015.
    """
    # Get the first and last day limits
    first_day = data.t.to_dict()[0].to_pydatetime()
    first_day = first_day.replace(hour=0, minute=0, second=0)
    last_day = data.t.to_dict()[data.shape[0] - 1].to_pydatetime()
    last_day = first_day.replace(day=last_day.day + 1, hour=0, minute=0, second=0)

    # Calculate the number of days and preallocate
    n_days = (last_day - first_day).days
    std_within = np.zeros(shape=(n_days,))

    for d in range(0, n_days):

        # Get the day of data
        low_limit = data.t >= (first_day + timedelta(days=d))
        high_limit = data.t < (first_day + timedelta(days=d + 1))
        flags = np.logical_and(low_limit, high_limit)
        day_data = data.glucose.values[flags]

        # Get daily mean and std
        std_within[d] = np.nanstd(day_data, ddof=1)

    # Compute index
    return np.nanmean(std_within)

def glucose_roc(data):
    """
    Computes the glucose rate-of-change (ROC) trace.
    As defined in the given reference, ROC at time t is defined as the difference
    between the glucose at time t and t-15 minutes divided by 15. By
    definition, the first two samples are always nan (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    glucose_roc: pd.Dataframe
        Dataframe with column `t` and `glucose_roc` containing the glucose ROC (in mg/dl/min).

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
    - Clarke et al., "Statistical Tools to Analyze Continuous Glucose
    Monitor Data", Diabetes Technol Ther, 2009,
    vol. 11, pp. S45-S54. DOI: 10.1089=dia.2008.0138.
    """
    g_roc = np.empty(shape=(data.glucose.values.size,))
    g_roc.fill(np.nan)

    if g_roc.size > 4:

        for t in range(3, g_roc.size):

            g_roc[t] = (data.glucose.values[t] - data.glucose.values[t-3]) / 15

    return pd.DataFrame(data={'t': data.t.values, 'glucose_roc': g_roc})


def std_glucose_roc(data):
    """
    Computes the standard of glucose rate-of-change of given data (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    std_glucose_roc: float
        The standard of glucose rate-of-change of given data.

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
    - Clarke et al., "Statistical Tools to Analyze Continuous Glucose
    Monitor Data", Diabetes Technol Ther, 2009,
    vol. 11, pp. S45-S54. DOI: 10.1089=dia.2008.0138.
    """
    roc = glucose_roc(data)

    return np.nanstd(roc.glucose_roc.values, ddof=1)


def cvga(data):
    """
    Computes the standard of glucose rate-of-change of given data (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    std_glucose_roc: float
        The standard of glucose rate-of-change of given data.

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
    - Magni et al., "Evaluating the efficacy of closed-loop glucose
    regulation via control-variability grid analysis", Journal of Diabetes
    Science and Technology, 2008, vol. 2, pp. 630-635. DOI:
    10.1177/193229680800200414.
    """
    roc = glucose_roc(data)

    x = np.min([np.max([110 - np.nanmin(data.glucose.values), 0]), 60])
    p = np.polyfit([110, 180, 300, 400], [0, 20, 40, 60], 3)
    y = np.polyval(p, np.nanmax(data.glucose.values))

    return x**2 + y**2
