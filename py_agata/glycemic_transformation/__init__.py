import numpy as np
import pandas as pd
from scipy.stats import iqr
from scipy.signal import find_peaks
from datetime import timedelta

from py_agata.time_in_ranges import time_in_target, time_in_hypoglycemia
from py_agata.input_validator import *

def mr_index(data, r=100):
    """
    Computes the mr value by Schlichtkrull (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)
    r: int, optional, default : 100
        Hyperparameter for mr value calculation

    Returns
    -------
    mr_index: float
        The mr value

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
    Schlichtkrull et al., "The M-value, an index of blood-sugar control in diabetics", Acta Medica Scandinavica, 1965,
    vol. 177, pp. 95-102. DOI: 10.1111/j.0954-6820.1965.tb01810.x
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)
    check_int_parameter(r)

    # Get non-nan values
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Return nan if all values are nan
    if values.size == 0:
        return np.nan

    # Return the result
    trans_data = 1000 * abs(np.log10(values / r))**3
    return np.mean(trans_data)


def hypo_index(data):
    """
    Computes the hypoglycemic index by Rodbard (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    hypo_index: float
        The hypo index

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
    Rodbard et al., "Interpretation of continuous glucose monitoring
    data: glycemic variability and quality of glycemic control", Diabetes
    %  Technology & Therapeutics, 2009, vol. 11, pp. S55-S67. DOI: 10.1089/dia.2008.0132.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    # Get non-nan values
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Return nan if all values are nan
    if values.size == 0:
        return np.nan

    # Set up the formula parameters
    b = 2
    d = 30
    lltr = 70

    # Compute metric
    return np.sum((lltr - values[values < lltr])**b) / (values.size * d)


def hyper_index(data):
    """
    Computes the hyperglycemic index by Rodbard (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    hyper_index: float
        The hyper index

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
    Rodbard et al., "Interpretation of continuous glucose monitoring
    data: glycemic variability and quality of glycemic control", Diabetes
    %  Technology & Therapeutics, 2009, vol. 11, pp. S55-S67. DOI: 10.1089/dia.2008.0132.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    # Get non-nan values
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Return nan if all values are nan
    if values.size == 0:
        return np.nan

    # Set up the formula parameters
    a = 1.1
    c = 30
    ultr = 180

    # Compute metric
    return np.sum((values[values > ultr] - ultr)**a) / (values.size * c)


def igc(data):
    """
    Computes the index of glycemic control by Rodbard (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    igc: float
        The index of glycemic control

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
    Rodbard et al., "Interpretation of continuous glucose monitoring
    data: glycemic variability and quality of glycemic control", Diabetes
    %  Technology & Therapeutics, 2009, vol. 11, pp. S55-S67. DOI: 10.1089/dia.2008.0132.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    # Compute metric
    return hypo_index(data) + hyper_index(data)


def grade_hypo_score(data):
    """
    Computes the glycemic risk assessment diabetes equation score in the hypoglycemic range (GRADEhypo)
    by Hill (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    grade_hypo_score: float
        The glycemic risk assessment diabetes equation score in the hypoglycemic range (GRADEhypo) (%).

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
    Hill et al., "A method for assessing quality of control from
    glucose profiles", Diabetic Medicine , 2007, vol. 24, pp. 753-758.
    DOI: 10.1111/j.1464-5491.2007.02119.x.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    # Get non-nan values
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Return nan if all values are nan
    if values.size == 0:
        return np.nan

    # Set up the formula parameters
    grade = 425 * (np.log10(np.log10(values / 18)) + .16)**2
    g_tot = np.sum(grade)
    return 100 * np.sum(grade[values < 70]) / g_tot


def grade_hyper_score(data):
    """
    Computes the glycemic risk assessment diabetes equation score in the hyperglycemic range (GRADEhyper)
    by Hill (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    grade_hyper_score: float
        The glycemic risk assessment diabetes equation score in the hyperglycemic range (GRADEhyper) (%).

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
    Hill et al., "A method for assessing quality of control from
    glucose profiles", Diabetic Medicine , 2007, vol. 24, pp. 753-758.
    DOI: 10.1111/j.1464-5491.2007.02119.x.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    # Get non-nan values
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Return nan if all values are nan
    if values.size == 0:
        return np.nan

    # Set up the formula parameters
    grade = 425 * (np.log10(np.log10(values / 18)) + .16)**2
    g_tot = np.sum(grade)
    return 100 * np.sum(grade[values > 180]) / g_tot


def grade_eu_score(data):
    """
    Computes the glycemic risk assessment diabetes equation score in the euglycemic range (GRADEeu)
    by Hill (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    grade_eu_score: float
        The glycemic risk assessment diabetes equation score in the euglycemic range (GRADEeu) (%).

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
    Hill et al., "A method for assessing quality of control from
    glucose profiles", Diabetic Medicine , 2007, vol. 24, pp. 753-758.
    DOI: 10.1111/j.1464-5491.2007.02119.x.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    return 100 - (grade_hypo_score(data) + grade_hyper_score(data))


def grade_score(data):
    """
    Computes the glycemic risk assessment diabetes equation score (GRADE)
    by Hill (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    grade_hyper_score: float
        The glycemic risk assessment diabetes equation score (GRADE) (%).

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
    Hill et al., "A method for assessing quality of control from
    glucose profiles", Diabetic Medicine , 2007, vol. 24, pp. 753-758.
    DOI: 10.1111/j.1464-5491.2007.02119.x.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    # Get non-nan values
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Return nan if all values are nan
    if values.size == 0:
        return np.nan

    # Set up the formula parameters
    grade = 425 * (np.log10(np.log10(values / 18)) + .16)**2
    return np.mean(grade)
