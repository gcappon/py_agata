import numpy as np


def time_in_target(data, glycemic_target='diabetes'):

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

    # Get non-nan values
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Get low/high flags
    flags_l = values >= th_l if include_th_l else values > th_l
    flags_h = values <= th_h if include_th_h else values < th_h

    # Return the results
    return 100 * np.where(np.logical_and(flags_l, flags_h))[0].shape[0]/values.shape[0]


def time_in_given_above_range(data, th, include_th=False):

    # Get non-nan values
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Get flags
    flags = values >= th if include_th else values > th

    # Return the results
    return 100 * np.where(flags)[0].shape[0]/values.shape[0]


def time_in_given_below_range(data, th, include_th=False):

    # Get non-nan values
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Get flags
    flags = values <= th if include_th else values < th

    # Return the results
    return 100 * np.where(flags)[0].shape[0]/values.shape[0]
