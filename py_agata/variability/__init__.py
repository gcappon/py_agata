import numpy as np
from scipy.stats import iqr
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
