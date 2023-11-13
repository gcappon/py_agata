import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.py_agata import Agata


def test_compare_two_arms():
    """
    Unit test of Agata.compare_two_arms function.

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
    data_1 = pd.DataFrame(data=d)

    t = np.arange(datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 1, 0, 55, 0), timedelta(minutes=5)).astype(
        datetime)
    glucose = np.zeros(shape=(t.shape[0],))
    glucose[0] = 40
    glucose[1:3] = [50, 60]
    glucose[3] = 120
    glucose[4:6] = [120, 150]
    glucose[6:8] = [190, 200]
    glucose[8:10] = [260, 260]
    glucose[10] = np.nan
    d = {'t': t, 'glucose': glucose}
    data_2 = pd.DataFrame(data=d)

    t = np.arange(datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 1, 0, 55, 0), timedelta(minutes=5)).astype(
        datetime)
    glucose = np.zeros(shape=(t.shape[0],))
    glucose[0] = 100
    glucose[1:3] = [100, 100]
    glucose[3] = 120
    glucose[4:6] = [120, 120]
    glucose[6:8] = [190, 200]
    glucose[8:10] = [100, 100]
    glucose[10] = np.nan
    d = {'t': t, 'glucose': glucose}
    data_3 = pd.DataFrame(data=d)

    t = np.arange(datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 1, 0, 55, 0), timedelta(minutes=5)).astype(
        datetime)
    glucose = np.zeros(shape=(t.shape[0],))
    glucose[0] = 100
    glucose[1:3] = [100, 100]
    glucose[3] = 120
    glucose[4:6] = [120, 120]
    glucose[6:8] = [100, 100]
    glucose[8:10] = [100, 100]
    glucose[10] = np.nan
    d = {'t': t, 'glucose': glucose}
    data_4 = pd.DataFrame(data=d)

    d = {'t': [], 'glucose': []}
    data_5 = pd.DataFrame(data=d)

    # Tests

    # Diabetes
    agata = Agata(glycemic_target='diabetes')

    results, stats = agata.compare_two_arms(arm_1=[data_1, data_1, data_2, data_2], arm_2=[data_3, data_3, data_4, data_4], is_paired=True, alpha=0.05)

    # Variability
    metric_list_name = ['mean_glucose', 'median_glucose', 'std_glucose', 'cv_glucose', 'range_glucose',
                        'iqr_glucose',
                        'auc_glucose', 'gmi', 'cogi', 'conga', 'j_index', 'mage_index', 'mage_minus_index',
                        'mage_plus_index',
                        'ef_index', 'modd', 'sddm_index', 'sdw_index', 'std_glucose_roc', 'cvga']
    for m in metric_list_name:
        assert results['arm_1']["variability"][m]["values"].size == 4
        assert results['arm_2']["variability"][m]["values"].size == 4
        assert stats["variability"][m]["h"] == 1 or stats["variability"][m]["h"] == 0 or np.isnan(stats["variability"][m]["h"])
        assert stats["variability"][m]["p"] >= 0 or stats["variability"][m]["p"] <= 1 or np.isnan(stats["variability"][m]["p"])

    # Time in ranges
    metric_list_name = ['time_in_target', 'time_in_tight_target', 'time_in_hypoglycemia',
                            'time_in_l1_hypoglycemia', 'time_in_l2_hypoglycemia', 'time_in_hyperglycemia',
                            'time_in_l1_hyperglycemia', 'time_in_l2_hyperglycemia']
    for m in metric_list_name:
        assert results['arm_1']["time_in_ranges"][m]["values"].size == 4
        assert results['arm_2']["time_in_ranges"][m]["values"].size == 4
        assert stats["time_in_ranges"][m]["h"] == 1 or stats["time_in_ranges"][m]["h"] == 0 or np.isnan(
            stats["time_in_ranges"][m]["h"])
        assert stats["time_in_ranges"][m]["p"] >= 0 or stats["time_in_ranges"][m]["p"] <= 1 or np.isnan(
            stats["time_in_ranges"][m]["p"])

    # Risk
    metric_list_name = ['adrr', 'lbgi', 'hbgi', 'bgri', 'gri']
    for m in metric_list_name:
        assert results['arm_1']["risk"][m]["values"].size == 4
        assert results['arm_2']["risk"][m]["values"].size == 4
        assert stats["risk"][m]["h"] == 1 or stats["risk"][m]["h"] == 0 or np.isnan(
            stats["risk"][m]["h"])
        assert stats["risk"][m]["p"] >= 0 or stats["risk"][m]["p"] <= 1 or np.isnan(
            stats["risk"][m]["p"])

    # Glycemic transformation
    metric_list_name = ['grade_score', 'grade_hypo_score', 'grade_hyper_score',
                            'grade_eu_score', 'igc', 'hypo_index','hyper_index',
                            'mr_index']
    for m in metric_list_name:
        assert results['arm_1']["glycemic_transformation"][m]["values"].size == 4
        assert results['arm_2']["glycemic_transformation"][m]["values"].size == 4
        assert stats["glycemic_transformation"][m]["h"] == 1 or stats["glycemic_transformation"][m]["h"] == 0 or np.isnan(
            stats["glycemic_transformation"][m]["h"])
        assert stats["glycemic_transformation"][m]["p"] >= 0 or stats["glycemic_transformation"][m]["p"] <= 1 or np.isnan(
            stats["glycemic_transformation"][m]["p"])

    # Data quality
    metric_list_name = ['number_days_of_observation', 'missing_glucose_percentage']
    for m in metric_list_name:
        assert results['arm_1']["data_quality"][m]["values"].size == 4
        assert results['arm_2']["data_quality"][m]["values"].size == 4
        assert stats["data_quality"][m]["h"] == 1 or stats["data_quality"][m][
            "h"] == 0 or np.isnan(
            stats["data_quality"][m]["h"])
        assert stats["data_quality"][m]["p"] >= 0 or stats["data_quality"][m][
            "p"] <= 1 or np.isnan(
            stats["data_quality"][m]["p"])

    # Events
    assert results["arm_1"]["events"]["hyperglycemic_events"]["hyper"]["mean_duration"]["values"].size == 4
    assert results["arm_1"]["events"]["hyperglycemic_events"]["hyper"]["mean_duration"]["values"].size == 4
    assert results["arm_1"]["events"]["hyperglycemic_events"]["hyper"]["events_per_week"]["values"].size == 4
    assert results["arm_1"]["events"]["hyperglycemic_events"]["l1"]["mean_duration"]["values"].size == 4
    assert results["arm_1"]["events"]["hyperglycemic_events"]["l1"]["events_per_week"]["values"].size == 4
    assert results["arm_1"]["events"]["hyperglycemic_events"]["l2"]["mean_duration"]["values"].size == 4
    assert results["arm_1"]["events"]["hyperglycemic_events"]["l2"]["events_per_week"]["values"].size == 4

    assert results["arm_1"]["events"]["hypoglycemic_events"]["hypo"]["mean_duration"]["values"].size == 4
    assert results["arm_1"]["events"]["hypoglycemic_events"]["hypo"]["events_per_week"]["values"].size == 4
    assert results["arm_1"]["events"]["hypoglycemic_events"]["l1"]["mean_duration"]["values"].size == 4
    assert results["arm_1"]["events"]["hypoglycemic_events"]["l1"]["events_per_week"]["values"].size == 4
    assert results["arm_1"]["events"]["hypoglycemic_events"]["l2"]["mean_duration"]["values"].size == 4
    assert results["arm_1"]["events"]["hypoglycemic_events"]["l2"]["events_per_week"]["values"].size == 4

    assert results["arm_1"]["events"]["extended_hypoglycemic_events"]["mean_duration"]["values"].size == 4
    assert  results["arm_1"]["events"]["extended_hypoglycemic_events"]["events_per_week"]["values"].size == 4

    assert results["arm_2"]["events"]["hyperglycemic_events"]["hyper"]["mean_duration"]["values"].size == 4
    assert results["arm_2"]["events"]["hyperglycemic_events"]["hyper"]["mean_duration"]["values"].size == 4
    assert results["arm_2"]["events"]["hyperglycemic_events"]["hyper"]["events_per_week"]["values"].size == 4
    assert results["arm_2"]["events"]["hyperglycemic_events"]["l1"]["mean_duration"]["values"].size == 4
    assert results["arm_2"]["events"]["hyperglycemic_events"]["l1"]["events_per_week"]["values"].size == 4
    assert results["arm_2"]["events"]["hyperglycemic_events"]["l2"]["mean_duration"]["values"].size == 4
    assert results["arm_2"]["events"]["hyperglycemic_events"]["l2"]["events_per_week"]["values"].size == 4

    assert results["arm_2"]["events"]["hypoglycemic_events"]["hypo"]["mean_duration"]["values"].size == 4
    assert results["arm_2"]["events"]["hypoglycemic_events"]["hypo"]["events_per_week"]["values"].size == 4
    assert results["arm_2"]["events"]["hypoglycemic_events"]["l1"]["mean_duration"]["values"].size == 4
    assert results["arm_2"]["events"]["hypoglycemic_events"]["l1"]["events_per_week"]["values"].size == 4
    assert results["arm_2"]["events"]["hypoglycemic_events"]["l2"]["mean_duration"]["values"].size == 4
    assert results["arm_2"]["events"]["hypoglycemic_events"]["l2"]["events_per_week"]["values"].size == 4

    assert results["arm_2"]["events"]["extended_hypoglycemic_events"]["mean_duration"]["values"].size == 4
    assert results["arm_2"]["events"]["extended_hypoglycemic_events"]["events_per_week"]["values"].size == 4


    results, stats = agata.compare_two_arms(arm_1=[data_1, data_1, data_2, data_2, data_4], arm_2=[data_3, data_3, data_4, data_4], is_paired=False, alpha=0.05)

    # Variability
    metric_list_name = ['mean_glucose', 'median_glucose', 'std_glucose', 'cv_glucose', 'range_glucose',
                        'iqr_glucose',
                        'auc_glucose', 'gmi', 'cogi', 'conga', 'j_index', 'mage_index', 'mage_minus_index',
                        'mage_plus_index',
                        'ef_index', 'modd', 'sddm_index', 'sdw_index', 'std_glucose_roc', 'cvga']
    for m in metric_list_name:
        assert results['arm_1']["variability"][m]["values"].size == 5
        assert results['arm_2']["variability"][m]["values"].size == 4
        assert stats["variability"][m]["h"] == 1 or stats["variability"][m]["h"] == 0 or np.isnan(stats["variability"][m]["h"])
        assert stats["variability"][m]["p"] >= 0 or stats["variability"][m]["p"] <= 1 or np.isnan(stats["variability"][m]["p"])

    # Time in ranges
    metric_list_name = ['time_in_target', 'time_in_tight_target', 'time_in_hypoglycemia',
                            'time_in_l1_hypoglycemia', 'time_in_l2_hypoglycemia', 'time_in_hyperglycemia',
                            'time_in_l1_hyperglycemia', 'time_in_l2_hyperglycemia']
    for m in metric_list_name:
        assert results['arm_1']["time_in_ranges"][m]["values"].size == 5
        assert results['arm_2']["time_in_ranges"][m]["values"].size == 4
        assert stats["time_in_ranges"][m]["h"] == 1 or stats["time_in_ranges"][m]["h"] == 0 or np.isnan(
            stats["time_in_ranges"][m]["h"])
        assert stats["time_in_ranges"][m]["p"] >= 0 or stats["time_in_ranges"][m]["p"] <= 1 or np.isnan(
            stats["time_in_ranges"][m]["p"])

    # Risk
    metric_list_name = ['adrr', 'lbgi', 'hbgi', 'bgri', 'gri']
    for m in metric_list_name:
        assert results['arm_1']["risk"][m]["values"].size == 5
        assert results['arm_2']["risk"][m]["values"].size == 4
        assert stats["risk"][m]["h"] == 1 or stats["risk"][m]["h"] == 0 or np.isnan(
            stats["risk"][m]["h"])
        assert stats["risk"][m]["p"] >= 0 or stats["risk"][m]["p"] <= 1 or np.isnan(
            stats["risk"][m]["p"])

    # Glycemic transformation
    metric_list_name = ['grade_score', 'grade_hypo_score', 'grade_hyper_score',
                            'grade_eu_score', 'igc', 'hypo_index','hyper_index',
                            'mr_index']
    for m in metric_list_name:
        assert results['arm_1']["glycemic_transformation"][m]["values"].size == 5
        assert results['arm_2']["glycemic_transformation"][m]["values"].size == 4
        assert stats["glycemic_transformation"][m]["h"] == 1 or stats["glycemic_transformation"][m]["h"] == 0 or np.isnan(
            stats["glycemic_transformation"][m]["h"])
        assert stats["glycemic_transformation"][m]["p"] >= 0 or stats["glycemic_transformation"][m]["p"] <= 1 or np.isnan(
            stats["glycemic_transformation"][m]["p"])

    # Data quality
    metric_list_name = ['number_days_of_observation', 'missing_glucose_percentage']
    for m in metric_list_name:
        assert results['arm_1']["data_quality"][m]["values"].size == 5
        assert results['arm_2']["data_quality"][m]["values"].size == 4
        assert stats["data_quality"][m]["h"] == 1 or stats["data_quality"][m][
            "h"] == 0 or np.isnan(
            stats["data_quality"][m]["h"])
        assert stats["data_quality"][m]["p"] >= 0 or stats["data_quality"][m][
            "p"] <= 1 or np.isnan(
            stats["data_quality"][m]["p"])

    # Events
    assert results["arm_1"]["events"]["hyperglycemic_events"]["hyper"]["mean_duration"]["values"].size == 5
    assert results["arm_1"]["events"]["hyperglycemic_events"]["hyper"]["mean_duration"]["values"].size == 5
    assert results["arm_1"]["events"]["hyperglycemic_events"]["hyper"]["events_per_week"]["values"].size == 5
    assert results["arm_1"]["events"]["hyperglycemic_events"]["l1"]["mean_duration"]["values"].size == 5
    assert results["arm_1"]["events"]["hyperglycemic_events"]["l1"]["events_per_week"]["values"].size == 5
    assert results["arm_1"]["events"]["hyperglycemic_events"]["l2"]["mean_duration"]["values"].size == 5
    assert results["arm_1"]["events"]["hyperglycemic_events"]["l2"]["events_per_week"]["values"].size == 5

    assert results["arm_1"]["events"]["hypoglycemic_events"]["hypo"]["mean_duration"]["values"].size == 5
    assert results["arm_1"]["events"]["hypoglycemic_events"]["hypo"]["events_per_week"]["values"].size == 5
    assert results["arm_1"]["events"]["hypoglycemic_events"]["l1"]["mean_duration"]["values"].size == 5
    assert results["arm_1"]["events"]["hypoglycemic_events"]["l1"]["events_per_week"]["values"].size == 5
    assert results["arm_1"]["events"]["hypoglycemic_events"]["l2"]["mean_duration"]["values"].size == 5
    assert results["arm_1"]["events"]["hypoglycemic_events"]["l2"]["events_per_week"]["values"].size == 5

    assert results["arm_1"]["events"]["extended_hypoglycemic_events"]["mean_duration"]["values"].size == 5
    assert  results["arm_1"]["events"]["extended_hypoglycemic_events"]["events_per_week"]["values"].size == 5

    assert results["arm_2"]["events"]["hyperglycemic_events"]["hyper"]["mean_duration"]["values"].size == 4
    assert results["arm_2"]["events"]["hyperglycemic_events"]["hyper"]["mean_duration"]["values"].size == 4
    assert results["arm_2"]["events"]["hyperglycemic_events"]["hyper"]["events_per_week"]["values"].size == 4
    assert results["arm_2"]["events"]["hyperglycemic_events"]["l1"]["mean_duration"]["values"].size == 4
    assert results["arm_2"]["events"]["hyperglycemic_events"]["l1"]["events_per_week"]["values"].size == 4
    assert results["arm_2"]["events"]["hyperglycemic_events"]["l2"]["mean_duration"]["values"].size == 4
    assert results["arm_2"]["events"]["hyperglycemic_events"]["l2"]["events_per_week"]["values"].size == 4

    assert results["arm_2"]["events"]["hypoglycemic_events"]["hypo"]["mean_duration"]["values"].size == 4
    assert results["arm_2"]["events"]["hypoglycemic_events"]["hypo"]["events_per_week"]["values"].size == 4
    assert results["arm_2"]["events"]["hypoglycemic_events"]["l1"]["mean_duration"]["values"].size == 4
    assert results["arm_2"]["events"]["hypoglycemic_events"]["l1"]["events_per_week"]["values"].size == 4
    assert results["arm_2"]["events"]["hypoglycemic_events"]["l2"]["mean_duration"]["values"].size == 4
    assert results["arm_2"]["events"]["hypoglycemic_events"]["l2"]["events_per_week"]["values"].size == 4

    assert results["arm_2"]["events"]["extended_hypoglycemic_events"]["mean_duration"]["values"].size == 4
    assert results["arm_2"]["events"]["extended_hypoglycemic_events"]["events_per_week"]["values"].size == 4

    results, stats = agata.compare_two_arms(arm_1=[data_5, data_5, data_5, data_5], arm_2=[data_5, data_5, data_5, data_5], is_paired=True, alpha=0.05)
    results, stats = agata.compare_two_arms(arm_1=[data_5, data_5, data_5, data_5], arm_2=[data_5, data_5, data_5], is_paired=False, alpha=0.05)

    results, stats = agata.compare_two_arms(arm_1=[data_1, data_1, data_1, data_1], arm_2=[data_1, data_1, data_1, data_1], is_paired=True, alpha=0.05)
