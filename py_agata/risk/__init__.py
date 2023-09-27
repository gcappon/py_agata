import numpy as np
from datetime import datetime,timedelta
from py_agata.time_in_ranges import time_in_l1_hypoglycemia, time_in_l2_hypoglycemia, time_in_l1_hyperglycemia, time_in_l2_hyperglycemia


def adrr(data):
    """
    Computes the average daily risk range (ADRR) of the glucose concentration (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl).

    Returns
    -------
    adrr: float
        the average daily risk range of the glucose concentration.

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
    Kovatchev et al., "Evaluation of a new measure of blood glucose variability in
    diabetes", Diabetes Care, 2006, vol. 29, pp. 2433-2438. DOI: 10.2337/dc06-1085.
    """
    # Setup the formula parameters
    alpha = 1.084
    beta = 5.381
    gamma = 1.509
    th = 112.5

    # Get the first and last day limits
    first_day = data.t.to_dict()[0].to_pydatetime()
    first_day = first_day.replace(hour=0, minute=0, second=0)
    last_day = data.t.to_dict()[data.shape[0]-1].to_pydatetime()
    last_day = first_day.replace(day=last_day.day+1,hour=0, minute=0, second=0)

    # Calculate the number of days and preallocate the daily max lbgi and hbgi
    n_days = (last_day-first_day).days
    max_lbgi_day = np.empty(shape=(n_days,))
    max_hbgi_day = np.empty(shape=(n_days,))

    for d in range(0,n_days):

        # Get the day of data
        low_limit = data.t >= (first_day + timedelta(days=d))
        high_limit = data.t < (first_day + timedelta(days=d+1))
        flags = np.logical_and(low_limit,high_limit)
        day_data = data.glucose.values[flags]

        # Get rid of nans
        non_nan_glucose = day_data[~np.isnan(day_data)]

        if not non_nan_glucose.size == 0:

            # Symmetrization
            f = gamma*(np.log(non_nan_glucose)**alpha-beta)

            # Risk computation
            rl = 10*(f**2)
            rl[non_nan_glucose > th] = 0
            rh = 10*(f**2)
            rh[non_nan_glucose < th] = 0

            # Get the max risks
            max_lbgi_day[d] = np.max(rl)
            max_hbgi_day[d] = np.max(rh)

        else:

            max_hbgi_day[d] = np.nan
            max_lbgi_day[d] = np.nan

    # Return adrr
    return np.nanmean(max_hbgi_day + max_lbgi_day)


def lbgi(data):
    """
    Computes the low blood glucose index (LBGI) of the glucose concentration (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl).

    Returns
    -------
    lbgi: float
        the low blood glucose index of the glucose concentration.

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
    Kovatchev et al., "Evaluation of a new measure of blood glucose variability in
    diabetes", Diabetes Care, 2006, vol. 29, pp. 2433-2438. DOI: 10.2337/dc06-1085.
    """
    # Setup the formula parameters
    alpha = 1.084
    beta = 5.381
    gamma = 1.509
    th = 112.5

    # Get rid of nans
    non_nan_glucose = data.glucose.values[~np.isnan(data.glucose.values)]

    # Symmetrization
    f = gamma*(np.log(non_nan_glucose)**alpha-beta)

    # Risk computation
    rl = 10*(f**2)
    rl[non_nan_glucose > th] = 0

    # Return lbgi
    return np.mean(rl)


def hbgi(data):
    """
    Computes the high blood glucose index (HBGI) of the glucose concentration (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl).

    Returns
    -------
    hbgi: float
        the high blood glucose index of the glucose concentration.

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
    Kovatchev et al., "Evaluation of a new measure of blood glucose variability in
    diabetes", Diabetes Care, 2006, vol. 29, pp. 2433-2438. DOI: 10.2337/dc06-1085.
    """
    # Setup the formula parameters
    alpha = 1.084
    beta = 5.381
    gamma = 1.509
    th = 112.5

    # Get rid of nans
    non_nan_glucose = data.glucose.values[~np.isnan(data.glucose.values)]

    # Symmetrization
    f = gamma*(np.log(non_nan_glucose)**alpha-beta)

    # Risk computation
    rh = 10*(f**2)
    rh[non_nan_glucose < th] = 0

    # Return hbgi
    return np.mean(rh)


def bgri(data):
    """
    Computes the blood glucose risk index (BGRI) of the glucose concentration (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl).

    Returns
    -------
    bgri: float
        the blood glucose risk index of the glucose concentration.

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
    Kovatchev et al., "Evaluation of a new measure of blood glucose variability in
    diabetes", Diabetes Care, 2006, vol. 29, pp. 2433-2438. DOI: 10.2337/dc06-1085.
    """

    # Return bgri
    return lbgi(data) + hbgi(data)


def gri(data):
    """
    Computes the blood glycemia risk index (GRI) of the glucose concentration (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl).

    Returns
    -------
    gri: float
        the glycemia risk index of the glucose concentration.

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
    Klonoff et al., "A Glycemia Risk Index (GRI) of hypoglycemia and
    hyperglycemia for continuous glucose monitoring validated by clinician
    ratings", Journal of Diabetes Science and Technology, 2022, pp. 1-17.
    DOI: 10.1177/19322968221085273.
    """

    #Compute metric
    v_low = time_in_l2_hypoglycemia(data) # VLow( < 54 mg / dL; < 3.0 mmol / L)
    low = time_in_l1_hypoglycemia(data) # Low(54–70 mg / dL; 3.0–3.9 mmol / L)
    v_high = time_in_l2_hyperglycemia(data) # VHigh( > 250 mg / dL; > 13.9 mmol / L)
    high = time_in_l1_hyperglycemia(data) # High( > 180–250 mg / dL; > 10.0–13.9 mmol / L)

    gri = (3.0 * v_low) + (2.4 * low) + (1.6 * v_high) + (0.8 * high)

    #Limit gri between 0 - 100 and return
    return np.min([gri, 100])


def dynamic_risk(data, amplification_function='tanh', maximum_amplification=2.5, amplification_rapidity=2, maximum_damping=0.6):
    """
    Computes the dynamic risk of the glucose concentration (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)
    amplification_function: str, {'exp', 'tanh'}, optional, default: 'tanh'
        Function to use to amplify the rate-of-change contribution to the dynamic risk
    maximum_amplification: float, optional, default: 2.5
        Intensity of amplification. Must be > 1
    amplification_rapidity: float, optional, default: 2
        Rapidity of the amplification. Must be > 0
    maximum_damping: float, optional, default: 0.6
        Damping of amplification. Must be > 0

    Returns
    -------
    dynamic_risk: float
        the dynamic risk of the glucose concentration.

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
    S. Guerra et al., "A Dynamic Risk Measure from Continuous Glucose Monitoring Data", Diabetes
    Technology & Therapeutics, 2011, vol. 13, pp. 843-852. DOI: 10.1089/dia.2011.0006
    """

    # If data is empty return null
    if data.shape[0] == 0:
        return np.nan

    # Initialization
    alpha = 1.084
    beta = 5.381
    gamma = 1.509
    dr_delta = (maximum_amplification - maximum_damping) / 2
    dr_beta = dr_delta + maximum_damping
    print((1 - dr_beta) / dr_delta)
    dr_gamma = np.arctanh(complex((1 - dr_beta) / dr_delta, 0))
    rl = np.zeros(shape=(data.glucose.values.size,))
    rh = np.zeros(shape=(data.glucose.values.size,))


    # Compute rate-of-change
    ts = (data.t.to_dict()[1].to_pydatetime() - data.t.to_dict()[0].to_pydatetime()).total_seconds()/60
    roc = np.diff(data.glucose.values)/ts
    roc = np.append(0,roc)

    # Symmetrization
    f = gamma * (np.log(data.glucose.values) ** alpha - beta)
    rl = 10 * (f ** 2)
    rl[f > 0] = 0
    rh = 10 * (f ** 2)
    rh[f < 0] = 0

    # Compute static risk
    sr = rh-rl
    modulation_factor = np.ones(shape=(data.glucose.values.size,))
    dr_over_dg = np.divide(10 * (gamma**2) * 2 * alpha * (np.log(data.glucose.values)**(2 * alpha - 1) - beta * np.log(data.glucose.values)**(alpha - 1)), data.glucose.values)

    # Compute dynamic risk and return it
    if amplification_function == 'tanh':
        modulation_factor = np.real(dr_delta*np.tanh(amplification_rapidity * np.multiply(dr_over_dg, roc) + dr_gamma)) + dr_beta
    elif amplification_function == 'exp':
        modulation_factor = np.exp(maximum_amplification * np.multiply(dr_over_dg, roc))
    return np.multiply(sr, modulation_factor)
