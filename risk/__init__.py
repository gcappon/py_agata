import numpy as np
from datetime import datetime,timedelta

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

        if ~(non_nan_glucose.size == 0):

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
