import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.py_agata import Agata


def test_analyze_glucose_profile():
    """
    Unit test of Agata.analyze_glucose_profile function.

    Parameters
    ----------
    None

    Returns
    -------
    None

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
    # Set test data
    t = np.arange(datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 1, 0, 55, 0), timedelta(minutes=5)).astype(
        datetime)
    glucose = np.zeros(shape=(t.shape[0],))
    glucose[0] = 40
    glucose[1:3] = [60, 60]
    glucose[3] = 80
    glucose[4:6] = [120, 150]
    glucose[6:8] = [200, 200]
    glucose[8:10] = [260, 260]
    glucose[10] = np.nan
    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    # Tests

    # Diabetes
    agata = Agata(data=data, glycemic_target='diabetes')

    results = agata.analyze_glucose_profile()
    assert type(results) is dict

    # Variability metrics
    assert type(results) is dict
    assert type(results['variability']) is dict
    assert np.round(results['variability']['mean_glucose'] * 1000) / 1000 == 143.000
    assert np.round(results['variability']['median_glucose'] * 1000) / 1000 == 135.000
    assert np.round(results['variability']['std_glucose'] * 1000) / 1000 == 83.540
    assert np.round(results['variability']['cv_glucose'] * 1000) / 1000 == 58.419
    assert np.round(results['variability']['range_glucose'] * 1000) / 1000 == 220.000
    assert np.round(results['variability']['iqr_glucose'] * 1000) / 1000 == 135.000
    assert np.round(results['variability']['auc_glucose'] * 1000) / 1000 == 7150.000
    assert np.round(results['variability']['gmi'] * 1000) / 1000 == 6.731
    assert np.round(results['variability']['cogi'] * 1000) / 1000 == 19.091
    assert np.isnan(np.round(results['variability']['conga'] * 1000) / 1000)
    assert np.round(results['variability']['j_index'] * 1000) / 1000 == 51.320
    assert np.round(results['variability']['mage_plus_index'] * 1000) / 1000 == 0.0
    assert np.round(results['variability']['mage_minus_index'] * 1000) / 1000 == 0.0
    assert np.round(results['variability']['mage_index'] * 1000) / 1000 == 0.0
    assert np.round(results['variability']['ef_index'] * 1000) / 1000 == 0.0
    assert np.isnan(np.round(results['variability']['modd'] * 1000) / 1000)
    assert np.isnan(np.round(results['variability']['sddm_index'] * 1000) / 1000)
    assert np.round(results['variability']['sdw_index'] * 1000) / 1000 == 83.540
    assert np.round(results['variability']['std_glucose_roc'] * 1000) / 1000 == 1.925
    assert np.round(results['variability']['cvga'] * 1000) / 1000 == 4759.436

    # Time fields
    assert type(results['time_in_ranges']) is dict
    assert results['time_in_ranges']['time_in_target'] == 30
    assert results['time_in_ranges']['time_in_tight_target'] == 20
    assert results['time_in_ranges']['time_in_hypoglycemia'] == 30
    assert results['time_in_ranges']['time_in_l1_hypoglycemia'] == 20
    assert results['time_in_ranges']['time_in_l2_hypoglycemia'] == 10
    assert results['time_in_ranges']['time_in_hyperglycemia'] == 40
    assert results['time_in_ranges']['time_in_l1_hyperglycemia'] == 20
    assert results['time_in_ranges']['time_in_l2_hyperglycemia'] == 20

    # Risk fields
    assert type(results['risk']) is dict
    assert np.round(results['risk']['adrr']*100)/100 == 61.13
    assert np.round(results['risk']['lbgi']*100)/100 == 6.76
    assert np.round(results['risk']['hbgi']*100)/100 == 7.57
    assert np.round(results['risk']['bgri']*100)/100 == 14.32
    assert np.round(results['risk']['gri']*100)/100 == 100.00

    # Pregnancy
    agata = Agata(data=data, glycemic_target='pregnancy')

    results = agata.analyze_glucose_profile()
    assert type(results) is dict

    # Variability metrics
    assert type(results['variability']) is dict
    assert np.round(results['variability']['mean_glucose']*1000)/1000 == 143.000
    assert np.round(results['variability']['median_glucose']*1000)/1000 == 135.000
    assert np.round(results['variability']['std_glucose']*1000)/1000 == 83.540
    assert np.round(results['variability']['cv_glucose']*1000)/1000 == 58.419
    assert np.round(results['variability']['range_glucose']*1000)/1000 == 220.000
    assert np.round(results['variability']['iqr_glucose']*1000)/1000 == 135.000
    assert np.round(results['variability']['auc_glucose']*1000)/1000 == 7150.000
    assert np.round(results['variability']['gmi']*1000)/1000 == 6.731
    assert np.round(results['variability']['cogi']*1000)/1000 == 19.091
    assert np.isnan(np.round(results['variability']['conga']*1000)/1000)
    assert np.round(results['variability']['j_index']*1000)/1000 == 51.320
    assert np.round(results['variability']['mage_plus_index']*1000)/1000 == 0.0
    assert np.round(results['variability']['mage_minus_index']*1000)/1000 == 0.0
    assert np.round(results['variability']['mage_index']*1000)/1000 == 0.0
    assert np.round(results['variability']['ef_index']*1000)/1000 == 0.0
    assert np.isnan(np.round(results['variability']['modd']*1000)/1000)
    assert np.isnan(np.round(results['variability']['sddm_index']*1000)/1000)
    assert np.round(results['variability']['sdw_index']*1000)/1000 == 83.540
    assert np.round(results['variability']['std_glucose_roc']*1000)/1000 == 1.925
    assert np.round(results['variability']['cvga']*1000)/1000 == 4759.436

    # Time fields
    assert type(results['time_in_ranges']) is dict
    assert results['time_in_ranges']['time_in_target'] == 20
    assert results['time_in_ranges']['time_in_tight_target'] == 20
    assert results['time_in_ranges']['time_in_hypoglycemia'] == 30
    assert results['time_in_ranges']['time_in_l1_hypoglycemia'] == 20
    assert results['time_in_ranges']['time_in_l2_hypoglycemia'] == 10
    assert results['time_in_ranges']['time_in_hyperglycemia'] == 50
    assert results['time_in_ranges']['time_in_l1_hyperglycemia'] == 30
    assert results['time_in_ranges']['time_in_l2_hyperglycemia'] == 20

    # Risk fields
    assert type(results['risk']) is dict
    assert np.round(results['risk']['adrr'] * 100) / 100 == 61.13
    assert np.round(results['risk']['lbgi'] * 100) / 100 == 6.76
    assert np.round(results['risk']['hbgi'] * 100) / 100 == 7.57
    assert np.round(results['risk']['bgri'] * 100) / 100 == 14.32
    assert np.round(results['risk']['gri'] * 100) / 100 == 100.00