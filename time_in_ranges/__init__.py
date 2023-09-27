import numpy as np


def time_in_target(data, glycemic_target='diabetes'):
    """
    Computes the time spent in the target range (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl).
    glycemic_target: str, {'diabetes', 'pregnancy'}, optional, default: 'diabetes'
        A string defining the set of glycemic targets to use. The default
        value is `diabetes`. It can be {`diabetes`,`pregnancy`).

    Returns
    -------
    time_in_target: float
        The time percentage spent in target range.

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
    Battelino et al., "Continuous glucose monitoring and metrics for clinical
    trials: An international consensus statement", The Lancet Diabetes &
    Endocrinology, 2022, pp. 1-16. DOI: https://doi.org/10.1016/S2213-8587(22)00319-9.
    """
    # Set the threshold
    if glycemic_target == 'diabetes':
        th_l = 70
        th_h = 180
    elif glycemic_target == 'pregnancy':
        th_l = 63
        th_h = 140
    else:
        raise RuntimeError('`glycemic_target` can be `diabetes` or `pregnancy`.')

    # Return the result
    return time_in_given_range(data=data, th_l=th_l, th_h=th_h, include_th_l=False, include_th_h=False)


def time_in_tight_target(data, glycemic_target='diabetes'):
    """
    Computes the time spent in the tight target range (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl).
    glycemic_target: str, {'diabetes', 'pregnancy'}, optional, default: 'diabetes'
        A string defining the set of glycemic targets to use. The default
        value is `diabetes`. It can be {`diabetes`,`pregnancy`).

    Returns
    -------
    time_in_tight_target: float
        The time percentage spent in tight target range.

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
    Battelino et al., "Continuous glucose monitoring and metrics for clinical
    trials: An international consensus statement", The Lancet Diabetes &
    Endocrinology, 2022, pp. 1-16. DOI: https://doi.org/10.1016/S2213-8587(22)00319-9.
    """
    # Set the threshold
    if glycemic_target == 'diabetes':
        th_l = 70
        th_h = 140
    elif glycemic_target == 'pregnancy':
        th_l = 70
        th_h = 140
    else:
        raise RuntimeError('`glycemic_target` can be `diabetes` or `pregnancy`.')

    # Return the result
    return time_in_given_range(data=data, th_l=th_l, th_h=th_h, include_th_l=False, include_th_h=False)


def time_in_hypoglycemia(data, glycemic_target='diabetes'):
    """
    Computes the time spent in the hypoglycemic range (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl).
    glycemic_target: str, {'diabetes', 'pregnancy'}, optional, default: 'diabetes'
        A string defining the set of glycemic targets to use. The default
        value is `diabetes`. It can be {`diabetes`,`pregnancy`).

    Returns
    -------
    time_in_hypoglycemia: float
        The time percentage spent in hypoglycemia.

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
    Battelino et al., "Continuous glucose monitoring and metrics for clinical
    trials: An international consensus statement", The Lancet Diabetes &
    Endocrinology, 2022, pp. 1-16. DOI: https://doi.org/10.1016/S2213-8587(22)00319-9.
    """
    # Set the threshold
    if glycemic_target == 'diabetes':
        th = 70
    elif glycemic_target == 'pregnancy':
        th = 63
    else:
        raise RuntimeError('`glycemic_target` can be `diabetes` or `pregnancy`.')

    # Return the result
    return time_in_given_below_range(data=data, th=th, include_th=True)


def time_in_l1_hypoglycemia(data, glycemic_target='diabetes'):
    """
    Computes the time spent in the l1 hypoglycemic range (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl).
    glycemic_target: str, {'diabetes', 'pregnancy'}, optional, default: 'diabetes'
        A string defining the set of glycemic targets to use. The default
        value is `diabetes`. It can be {`diabetes`,`pregnancy`).

    Returns
    -------
    time_in_l1_hypoglycemia: float
        The time percentage spent in l1 hypoglycemia.

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
    Battelino et al., "Continuous glucose monitoring and metrics for clinical
    trials: An international consensus statement", The Lancet Diabetes &
    Endocrinology, 2022, pp. 1-16. DOI: https://doi.org/10.1016/S2213-8587(22)00319-9.
    """
    # Set the threshold
    if glycemic_target == 'diabetes':
        th_l = 54
        th_h = 70
    elif glycemic_target == 'pregnancy':
        th_l = 54
        th_h = 63
    else:
        raise RuntimeError('`glycemic_target` can be `diabetes` or `pregnancy`.')

    # Return the result
    return time_in_given_range(data=data, th_l=th_l, th_h=th_h, include_th_l=False, include_th_h=True)


def time_in_l2_hypoglycemia(data, glycemic_target='diabetes'):
    """
    Computes the time spent in the l2 hypoglycemic range (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl).
    glycemic_target: str, {'diabetes', 'pregnancy'}, optional, default: 'diabetes'
        A string defining the set of glycemic targets to use. The default
        value is `diabetes`. It can be {`diabetes`,`pregnancy`).

    Returns
    -------
    time_in_l2_hypoglycemia: float
        The time percentage spent in l2 hypoglycemia.

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
    Battelino et al., "Continuous glucose monitoring and metrics for clinical
    trials: An international consensus statement", The Lancet Diabetes &
    Endocrinology, 2022, pp. 1-16. DOI: https://doi.org/10.1016/S2213-8587(22)00319-9.
    """
    # Set the threshold
    if glycemic_target == 'diabetes':
        th = 54
    elif glycemic_target == 'pregnancy':
        th = 54
    else:
        raise RuntimeError('`glycemic_target` can be `diabetes` or `pregnancy`.')

    # Return the result
    return time_in_given_below_range(data=data, th=th, include_th=True)


def time_in_hyperglycemia(data, glycemic_target='diabetes'):
    """
    Computes the time spent in the hyperglycemic range (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl).
    glycemic_target: str, {'diabetes', 'pregnancy'}, optional, default: 'diabetes'
        A string defining the set of glycemic targets to use. The default
        value is `diabetes`. It can be {`diabetes`,`pregnancy`).

    Returns
    -------
    time_in_hyperglycemia: float
        The time percentage spent in hyperglycemia.

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
    Battelino et al., "Continuous glucose monitoring and metrics for clinical
    trials: An international consensus statement", The Lancet Diabetes &
    Endocrinology, 2022, pp. 1-16. DOI: https://doi.org/10.1016/S2213-8587(22)00319-9.
    """
    # Set the threshold
    if glycemic_target == 'diabetes':
        th = 180
    elif glycemic_target == 'pregnancy':
        th = 140
    else:
        raise RuntimeError('`glycemic_target` can be `diabetes` or `pregnancy`.')

    # Return the result
    return time_in_given_above_range(data=data, th=th, include_th=True)


def time_in_l1_hyperglycemia(data, glycemic_target='diabetes'):
    """
    Computes the time spent in the l1 hyperglycemic range (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl).
    glycemic_target: str, {'diabetes', 'pregnancy'}, optional, default: 'diabetes'
        A string defining the set of glycemic targets to use. The default
        value is `diabetes`. It can be {`diabetes`,`pregnancy`).

    Returns
    -------
    time_in_l1_hyperglycemia: float
        The time percentage spent in l1 hyperglycemia.

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
    Battelino et al., "Continuous glucose monitoring and metrics for clinical
    trials: An international consensus statement", The Lancet Diabetes &
    Endocrinology, 2022, pp. 1-16. DOI: https://doi.org/10.1016/S2213-8587(22)00319-9.
    """
    # Set the threshold
    if glycemic_target == 'diabetes':
        th_l = 180
        th_h = 250
    elif glycemic_target == 'pregnancy':
        th_l = 140
        th_h = 250
    else:
        raise RuntimeError('`glycemic_target` can be `diabetes` or `pregnancy`.')

    # Return the result
    return time_in_given_range(data=data, th_l=th_l, th_h=th_h, include_th_l=True, include_th_h=False)


def time_in_l2_hyperglycemia(data, glycemic_target='diabetes'):
    """
    Computes the time spent in the l2 hyperglycemic range (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl).
    glycemic_target: str, {'diabetes', 'pregnancy'}, optional, default: 'diabetes'
        A string defining the set of glycemic targets to use. The default
        value is `diabetes`. It can be {`diabetes`,`pregnancy`).

    Returns
    -------
    time_in_l2_hyperglycemia: float
        The time percentage spent in l2 hyperglycemia.

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
    Battelino et al., "Continuous glucose monitoring and metrics for clinical
    trials: An international consensus statement", The Lancet Diabetes &
    Endocrinology, 2022, pp. 1-16. DOI: https://doi.org/10.1016/S2213-8587(22)00319-9.
    """
    # Set the threshold
    if glycemic_target == 'diabetes':
        th = 250
    elif glycemic_target == 'pregnancy':
        th = 250
    else:
        raise RuntimeError('`glycemic_target` can be `diabetes` or `pregnancy`.')

    # Return the result
    return time_in_given_above_range(data=data, th=th, include_th=True)


def time_in_given_range(data, th_l, th_h, include_th_l=False, include_th_h=False):
    """
    Computes the time spent between a given range (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl).
    th_l: float
        The low level threshold of the range of interest (in mg/dl).
    th_h: float
        The high level threshold of the range of interest (in mg/dl).
    include_th_l: bool, optional, default: False
        A flag indicating whether to include or not th_l in the range of interest.
    include_th_h: bool, optional, default: False
        A flag indicating whether to include or not th_h in the range of interest.

    Returns
    -------
    time_in_given_range: float
        The time percentage spent in the given range.

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
    Battelino et al., "Continuous glucose monitoring and metrics for clinical
    trials: An international consensus statement", The Lancet Diabetes &
    Endocrinology, 2022, pp. 1-16. DOI: https://doi.org/10.1016/S2213-8587(22)00319-9.
    """
    # Get non-nan values
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Return nan if all values are nan
    if values.size == 0:
        return np.nan

    # Get low/high flags
    flags_l = values >= th_l if include_th_l else values > th_l
    flags_h = values <= th_h if include_th_h else values < th_h

    # Return the results
    return 100 * np.where(np.logical_and(flags_l, flags_h))[0].shape[0]/values.shape[0]


def time_in_given_above_range(data, th, include_th=False):
    """
    Computes the time spent above a given range (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl).
    th: float
        The threshold of the range of interest (in mg/dl).
    include_th: bool, optional, default: False
        A flag indicating whether to include or not th in the range of interest.

    Returns
    -------
    time_in_given_above_range: float
        The time percentage spent above the given range.

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
    Battelino et al., "Continuous glucose monitoring and metrics for clinical
    trials: An international consensus statement", The Lancet Diabetes &
    Endocrinology, 2022, pp. 1-16. DOI: https://doi.org/10.1016/S2213-8587(22)00319-9.
    """
    # Get non-nan values
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Return nan if all values are nan
    if values.size == 0:
        return np.nan

    # Get flags
    flags = values >= th if include_th else values > th

    # Return the results
    return 100 * np.where(flags)[0].shape[0]/values.shape[0]


def time_in_given_below_range(data, th, include_th=False):
    """
    Computes the time spent below a given range (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl).
    th: float
        The threshold of the range of interest (in mg/dl).
    include_th: bool, optional, default: False
        A flag indicating whether to include or not th in the range of interest.

    Returns
    -------
    time_in_given_below_range: float
        The time percentage spent below the given range.

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
    Battelino et al., "Continuous glucose monitoring and metrics for clinical
    trials: An international consensus statement", The Lancet Diabetes &
    Endocrinology, 2022, pp. 1-16. DOI: https://doi.org/10.1016/S2213-8587(22)00319-9.
    """
    # Get non-nan values
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Return nan if all values are nan
    if values.size == 0:
        return np.nan

    # Get flags
    flags = values <= th if include_th else values < th

    # Return the results
    return 100 * np.where(flags)[0].shape[0]/values.shape[0]
