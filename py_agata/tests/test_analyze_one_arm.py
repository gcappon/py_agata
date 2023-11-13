import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.py_agata import Agata


def test_analyze_one_arm():
    """
    Unit test of Agata.analyze_one_arm function.

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
    agata = Agata(glycemic_target='diabetes')

    results = agata.analyze_one_arm(data=[data,data])

    # Variability
    metric_list_name = ['mean_glucose', 'median_glucose', 'std_glucose', 'cv_glucose', 'range_glucose',
                        'iqr_glucose',
                        'auc_glucose', 'gmi', 'cogi', 'conga', 'j_index', 'mage_index', 'mage_minus_index',
                        'mage_plus_index',
                        'ef_index', 'modd', 'sddm_index', 'sdw_index', 'std_glucose_roc', 'cvga']
    for m in metric_list_name:
        assert results["variability"][m]["values"].size == 2

    # Time in ranges
    metric_list_name = ['time_in_target', 'time_in_tight_target', 'time_in_hypoglycemia',
                            'time_in_l1_hypoglycemia', 'time_in_l2_hypoglycemia', 'time_in_hyperglycemia',
                            'time_in_l1_hyperglycemia', 'time_in_l2_hyperglycemia']
    for m in metric_list_name:
        assert results["time_in_ranges"][m]["values"].size == 2

    # Risk
    metric_list_name = ['adrr', 'lbgi', 'hbgi', 'bgri', 'gri']
    for m in metric_list_name:
        assert results["risk"][m]["values"].size == 2

    # Glycemic transformation
    metric_list_name = ['grade_score', 'grade_hypo_score', 'grade_hyper_score',
                            'grade_eu_score', 'igc', 'hypo_index','hyper_index',
                            'mr_index']
    for m in metric_list_name:
        assert results["glycemic_transformation"][m]["values"].size == 2

    # Data quality
    metric_list_name = ['number_days_of_observation', 'missing_glucose_percentage']
    for m in metric_list_name:
        assert results["data_quality"][m]["values"].size == 2

    # Events
    assert results["events"]["hyperglycemic_events"]["hyper"]["mean_duration"]["values"].size == 2
    assert results["events"]["hyperglycemic_events"]["hyper"]["events_per_week"]["values"].size == 2
    assert results["events"]["hyperglycemic_events"]["l1"]["mean_duration"]["values"].size == 2
    assert results["events"]["hyperglycemic_events"]["l1"]["events_per_week"]["values"].size == 2
    assert results["events"]["hyperglycemic_events"]["l2"]["mean_duration"]["values"].size == 2
    assert results["events"]["hyperglycemic_events"]["l2"]["events_per_week"]["values"].size == 2

    assert results["events"]["hypoglycemic_events"]["hypo"]["mean_duration"]["values"].size == 2
    assert results["events"]["hypoglycemic_events"]["hypo"]["events_per_week"]["values"].size == 2
    assert results["events"]["hypoglycemic_events"]["l1"]["mean_duration"]["values"].size == 2
    assert results["events"]["hypoglycemic_events"]["l1"]["events_per_week"]["values"].size == 2
    assert results["events"]["hypoglycemic_events"]["l2"]["mean_duration"]["values"].size == 2
    assert results["events"]["hypoglycemic_events"]["l2"]["events_per_week"]["values"].size == 2

    assert results["events"]["extended_hypoglycemic_events"]["mean_duration"]["values"].size == 2
    assert  results["events"]["extended_hypoglycemic_events"]["events_per_week"]["values"].size == 2