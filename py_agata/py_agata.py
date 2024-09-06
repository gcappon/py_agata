import numpy as np
from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu,ranksums

from py_agata.variability import *
from py_agata.time_in_ranges import *
from py_agata.risk import *
from py_agata.glycemic_transformation import *
from py_agata.inspection import *

class Agata:
    """
    Core class of AGATA.

    ...
    Attributes
    ----------
    glycemic_target: str
        A string defining the set of glycemic targets to use.

    Methods
    -------
    analyze_glucose_profile():
        Runs ReplayBG.
    """

    def __init__(self, glycemic_target='diabetes'):
        self.glycemic_target = glycemic_target

    def analyze_glucose_profile(self, data):
        """
        Analyzes a single glucose profile.

        Parameters
        ----------
        data: pd.DataFrame
            Pandas dataframe with a column `glucose` containing the glucose data to analyze (in mg/dl).

        Returns
        -------
        results: dict
            A dictionary containing the results of the analysis i.e.:
            - variability: dict
                A dictionary containing the values of the variability related metrics.
            - time_in_ranges: dict
                A dictionary containing the values of the time in range related metrics.
            - risk: dict
                A dictionary containing the values of the risk related metrics.
            - glycemic_transformation: dict
                A dictionary containing the values of the glycemic_transformation related metrics.
            - data_quality: dict
                A dictionary containing the values of the data quality related metrics.
            - events: dict
                A dictionary containing the values of the events related metrics.


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
        # Check input
        check_dataframe(data)
        check_data_columns(data)
        check_homogeneous_timegrid(data)

        results = dict()

        # Get variability metrics
        results['variability'] = dict()
        results['variability']['mean_glucose'] = mean_glucose(data)
        results['variability']['median_glucose'] = median_glucose(data)
        results['variability']['std_glucose'] = std_glucose(data)
        results['variability']['cv_glucose'] = cv_glucose(data)
        results['variability']['range_glucose'] = range_glucose(data)
        results['variability']['iqr_glucose'] = iqr_glucose(data)
        results['variability']['auc_glucose'] = auc_glucose(data)
        results['variability']['gmi'] = gmi(data)
        results['variability']['cogi'] = cogi(data)
        results['variability']['conga'] = conga(data)
        results['variability']['j_index'] = j_index(data)
        results['variability']['mage_plus_index'] = mage_plus_index(data)
        results['variability']['mage_minus_index'] = mage_minus_index(data)
        results['variability']['mage_index'] = mage_index(data)
        results['variability']['ef_index'] = ef_index(data)
        results['variability']['modd'] = modd(data)
        results['variability']['sddm_index'] = sddm_index(data)
        results['variability']['sdw_index'] = sdw_index(data)
        results['variability']['std_glucose_roc'] = std_glucose_roc(data)
        results['variability']['cvga'] = cvga(data)

        # Get time metrics
        results['time_in_ranges'] = dict()
        results['time_in_ranges']['time_in_target'] = time_in_target(data, self.glycemic_target)
        results['time_in_ranges']['time_in_tight_target'] = time_in_tight_target(data, self.glycemic_target)
        results['time_in_ranges']['time_in_hypoglycemia'] = time_in_hypoglycemia(data, self.glycemic_target)
        results['time_in_ranges']['time_in_l1_hypoglycemia'] = time_in_l1_hypoglycemia(data, self.glycemic_target)
        results['time_in_ranges']['time_in_l2_hypoglycemia'] = time_in_l2_hypoglycemia(data, self.glycemic_target)
        results['time_in_ranges']['time_in_hyperglycemia'] = time_in_hyperglycemia(data, self.glycemic_target)
        results['time_in_ranges']['time_in_l1_hyperglycemia'] = time_in_l1_hyperglycemia(data, self.glycemic_target)
        results['time_in_ranges']['time_in_l2_hyperglycemia'] = time_in_l2_hyperglycemia(data, self.glycemic_target)

        # Get risk metrics
        results['risk'] = dict()
        results['risk']['adrr'] = adrr(data)
        results['risk']['lbgi'] = lbgi(data)
        results['risk']['hbgi'] = hbgi(data)
        results['risk']['bgri'] = bgri(data)
        results['risk']['gri'] = gri(data)

        # Get glycemic transformation metrics
        results['glycemic_transformation'] = dict()
        results['glycemic_transformation']['grade_score'] = grade_score(data)
        results['glycemic_transformation']['grade_hypo_score'] = grade_hypo_score(data)
        results['glycemic_transformation']['grade_hyper_score'] = grade_hyper_score(data)
        results['glycemic_transformation']['grade_eu_score'] = grade_eu_score(data)
        results['glycemic_transformation']['igc'] = igc(data)
        results['glycemic_transformation']['hypo_index'] = hypo_index(data)
        results['glycemic_transformation']['hyper_index'] = hyper_index(data)
        results['glycemic_transformation']['mr_index'] = mr_index(data)

        # Event metrics
        results['events'] = dict()
        results['events']['hypoglycemic_events'] = find_hypoglycemic_events_by_level(data, glycemic_target=self.glycemic_target)
        results['events']['hyperglycemic_events'] = find_hyperglycemic_events_by_level(data, glycemic_target=self.glycemic_target)
        results['events']['extended_hypoglycemic_events'] = find_extended_hypoglycemic_events(data)

        # Data quality metrics
        results['data_quality'] = dict()
        results['data_quality']['number_days_of_observation'] = number_days_of_observation(data)
        results['data_quality']['missing_glucose_percentage'] = missing_glucose_percentage(data)

        # Return results
        return results

    def analyze_one_arm(self, data):
        """
        Analyzes glucose data of one arm.

        Parameters
        ----------
        data: list of pd.DataFrame
            List of pandas dataframes with a column `glucose` containing the glucose data
            to analyze (in mg/dl).

        Returns
        -------
        results: dict
            A dictionary containing the results of the analysis i.e.:
            - variability: dict
                A dictionary containing the values of the variability related metrics.
            - time_in_ranges: dict
                A dictionary containing the values of the time in range related metrics.
            - risk: dict
                A dictionary containing the values of the risk related metrics.
            - glycemic_transformation: dict
                A dictionary containing the values of the glycemic_transformation related metrics.
            - data_quality: dict
                A dictionary containing the values of the data quality related metrics.
            - events: dict
                A dictionary containing the values of the events related metrics.


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
        for d in data:
            # Check input
            check_dataframe(d)
            check_data_columns(d)
            check_homogeneous_timegrid(d)

        results = dict()


        # Variability
        results["variability"] = dict()
        metric_list = [mean_glucose, median_glucose, std_glucose, cv_glucose, range_glucose, iqr_glucose,
                       auc_glucose, gmi, cogi, conga, j_index, mage_index, mage_minus_index, mage_plus_index,
                       ef_index, modd, sddm_index, sdw_index, std_glucose_roc, cvga]
        metric_list_name = ['mean_glucose', 'median_glucose', 'std_glucose', 'cv_glucose', 'range_glucose',
                            'iqr_glucose',
                            'auc_glucose', 'gmi', 'cogi', 'conga', 'j_index', 'mage_index', 'mage_minus_index',
                            'mage_plus_index',
                            'ef_index', 'modd', 'sddm_index', 'sdw_index', 'std_glucose_roc', 'cvga']

        for m in range(len(metric_list_name)):
            results["variability"][metric_list_name[m]] = dict()
            results["variability"][metric_list_name[m]]["values"] = np.zeros(shape=(len(data),))
            for d in range(len(data)):
                results["variability"][metric_list_name[m]]["values"][d] = metric_list[m](data[d])
            results["variability"][metric_list_name[m]]["mean"] = np.nanmean(results["variability"][metric_list_name[m]]["values"])
            results["variability"][metric_list_name[m]]["std"] = np.nanstd(results["variability"][metric_list_name[m]]["values"])
            results["variability"][metric_list_name[m]]["median"] = np.nanmedian(results["variability"][metric_list_name[m]]["values"])
            results["variability"][metric_list_name[m]]["prc_5"] = np.nanpercentile(results["variability"][metric_list_name[m]]["values"], 5)
            results["variability"][metric_list_name[m]]["prc_25"] = np.nanpercentile(results["variability"][metric_list_name[m]]["values"], 25)
            results["variability"][metric_list_name[m]]["prc_75"] = np.nanpercentile(results["variability"][metric_list_name[m]]["values"], 75)
            results["variability"][metric_list_name[m]]["prc_95"] = np.nanpercentile(results["variability"][metric_list_name[m]]["values"], 95)


        # Time in ranges
        results["time_in_ranges"] = dict()
        metric_list = [time_in_target, time_in_tight_target, time_in_hypoglycemia,
                       time_in_l1_hypoglycemia, time_in_l2_hypoglycemia, time_in_hyperglycemia,
                       time_in_l1_hyperglycemia, time_in_l2_hyperglycemia]
        metric_list_name = ['time_in_target', 'time_in_tight_target', 'time_in_hypoglycemia',
                            'time_in_l1_hypoglycemia', 'time_in_l2_hypoglycemia', 'time_in_hyperglycemia',
                            'time_in_l1_hyperglycemia', 'time_in_l2_hyperglycemia']

        for m in range(len(metric_list_name)):
            results["time_in_ranges"][metric_list_name[m]] = dict()
            results["time_in_ranges"][metric_list_name[m]]["values"] = np.zeros(shape=(len(data),))
            for d in range(len(data)):
                results["time_in_ranges"][metric_list_name[m]]["values"][d] = metric_list[m](data[d], self.glycemic_target)
            results["time_in_ranges"][metric_list_name[m]]["mean"] = np.nanmean(
                results["time_in_ranges"][metric_list_name[m]]["values"])
            results["time_in_ranges"][metric_list_name[m]]["std"] = np.nanstd(
                results["time_in_ranges"][metric_list_name[m]]["values"])
            results["time_in_ranges"][metric_list_name[m]]["median"] = np.nanmedian(
                results["time_in_ranges"][metric_list_name[m]]["values"])
            results["time_in_ranges"][metric_list_name[m]]["prc_5"] = np.nanpercentile(
                results["time_in_ranges"][metric_list_name[m]]["values"], 5)
            results["time_in_ranges"][metric_list_name[m]]["prc_25"] = np.nanpercentile(
                results["time_in_ranges"][metric_list_name[m]]["values"], 25)
            results["time_in_ranges"][metric_list_name[m]]["prc_75"] = np.nanpercentile(
                results["time_in_ranges"][metric_list_name[m]]["values"], 75)
            results["time_in_ranges"][metric_list_name[m]]["prc_95"] = np.nanpercentile(
                results["time_in_ranges"][metric_list_name[m]]["values"], 95)

        # Risk
        results["risk"] = dict()
        metric_list = [adrr, lbgi, hbgi, bgri, gri]
        metric_list_name = ['adrr', 'lbgi', 'hbgi', 'bgri', 'gri']

        for m in range(len(metric_list_name)):
            results["risk"][metric_list_name[m]] = dict()
            results["risk"][metric_list_name[m]]["values"] = np.zeros(shape=(len(data),))
            for d in range(len(data)):
                results["risk"][metric_list_name[m]]["values"][d] = metric_list[m](data[d])
            results["risk"][metric_list_name[m]]["mean"] = np.nanmean(
                results["risk"][metric_list_name[m]]["values"])
            results["risk"][metric_list_name[m]]["std"] = np.nanstd(
                results["risk"][metric_list_name[m]]["values"])
            results["risk"][metric_list_name[m]]["median"] = np.nanmedian(
                results["risk"][metric_list_name[m]]["values"])
            results["risk"][metric_list_name[m]]["prc_5"] = np.nanpercentile(
                results["risk"][metric_list_name[m]]["values"], 5)
            results["risk"][metric_list_name[m]]["prc_25"] = np.nanpercentile(
                results["risk"][metric_list_name[m]]["values"], 25)
            results["risk"][metric_list_name[m]]["prc_75"] = np.nanpercentile(
                results["risk"][metric_list_name[m]]["values"], 75)
            results["risk"][metric_list_name[m]]["prc_95"] = np.nanpercentile(
                results["risk"][metric_list_name[m]]["values"], 95)

        # Glycemic transformation
        results["glycemic_transformation"] = dict()
        metric_list = [grade_score, grade_hypo_score, grade_hyper_score,
                       grade_eu_score, igc, hypo_index,
                       hyper_index, mr_index]
        metric_list_name = ['grade_score', 'grade_hypo_score', 'grade_hyper_score',
                            'grade_eu_score', 'igc', 'hypo_index','hyper_index',
                            'mr_index']

        for m in range(len(metric_list_name)):
            results["glycemic_transformation"][metric_list_name[m]] = dict()
            results["glycemic_transformation"][metric_list_name[m]]["values"] = np.zeros(shape=(len(data),))
            for d in range(len(data)):
                results["glycemic_transformation"][metric_list_name[m]]["values"][d] = metric_list[m](data[d])
            results["glycemic_transformation"][metric_list_name[m]]["mean"] = np.nanmean(
                results["glycemic_transformation"][metric_list_name[m]]["values"])
            results["glycemic_transformation"][metric_list_name[m]]["std"] = np.nanstd(
                results["glycemic_transformation"][metric_list_name[m]]["values"])
            results["glycemic_transformation"][metric_list_name[m]]["median"] = np.nanmedian(
                results["glycemic_transformation"][metric_list_name[m]]["values"])
            results["glycemic_transformation"][metric_list_name[m]]["prc_5"] = np.nanpercentile(
                results["glycemic_transformation"][metric_list_name[m]]["values"], 5)
            results["glycemic_transformation"][metric_list_name[m]]["prc_25"] = np.nanpercentile(
                results["glycemic_transformation"][metric_list_name[m]]["values"], 25)
            results["glycemic_transformation"][metric_list_name[m]]["prc_75"] = np.nanpercentile(
                results["glycemic_transformation"][metric_list_name[m]]["values"], 75)
            results["glycemic_transformation"][metric_list_name[m]]["prc_95"] = np.nanpercentile(
                results["glycemic_transformation"][metric_list_name[m]]["values"], 95)

        # Data quality
        results["data_quality"] = dict()
        metric_list = [number_days_of_observation, missing_glucose_percentage]
        metric_list_name = ['number_days_of_observation','missing_glucose_percentage']

        for m in range(len(metric_list_name)):
            results["data_quality"][metric_list_name[m]] = dict()
            results["data_quality"][metric_list_name[m]]["values"] = np.zeros(shape=(len(data),))
            for d in range(len(data)):
                results["data_quality"][metric_list_name[m]]["values"][d] = metric_list[m](data[d])
            results["data_quality"][metric_list_name[m]]["mean"] = np.nanmean(
                results["data_quality"][metric_list_name[m]]["values"])
            results["data_quality"][metric_list_name[m]]["std"] = np.nanstd(
                results["data_quality"][metric_list_name[m]]["values"])
            results["data_quality"][metric_list_name[m]]["median"] = np.nanmedian(
                results["data_quality"][metric_list_name[m]]["values"])
            results["data_quality"][metric_list_name[m]]["prc_5"] = np.nanpercentile(
                results["data_quality"][metric_list_name[m]]["values"], 5)
            results["data_quality"][metric_list_name[m]]["prc_25"] = np.nanpercentile(
                results["data_quality"][metric_list_name[m]]["values"], 25)
            results["data_quality"][metric_list_name[m]]["prc_75"] = np.nanpercentile(
                results["data_quality"][metric_list_name[m]]["values"], 75)
            results["data_quality"][metric_list_name[m]]["prc_95"] = np.nanpercentile(
                results["data_quality"][metric_list_name[m]]["values"], 95)

        # Events
        results["events"] = dict()
        results["events"]["hyperglycemic_events"] = dict()
        results["events"]["hyperglycemic_events"]["hyper"] = dict()
        results["events"]["hyperglycemic_events"]["hyper"]["mean_duration"] = dict()
        results["events"]["hyperglycemic_events"]["hyper"]["mean_duration"]["values"] = np.zeros(shape=(len(data),))
        results["events"]["hyperglycemic_events"]["hyper"]["events_per_week"] = dict()
        results["events"]["hyperglycemic_events"]["hyper"]["events_per_week"]["values"] = np.zeros(shape=(len(data),))
        results["events"]["hyperglycemic_events"]["l1"] = dict()
        results["events"]["hyperglycemic_events"]["l1"]["mean_duration"] = dict()
        results["events"]["hyperglycemic_events"]["l1"]["mean_duration"]["values"] = np.zeros(shape=(len(data),))
        results["events"]["hyperglycemic_events"]["l1"]["events_per_week"] = dict()
        results["events"]["hyperglycemic_events"]["l1"]["events_per_week"]["values"] = np.zeros(shape=(len(data),))
        results["events"]["hyperglycemic_events"]["l2"] = dict()
        results["events"]["hyperglycemic_events"]["l2"]["mean_duration"] = dict()
        results["events"]["hyperglycemic_events"]["l2"]["mean_duration"]["values"] = np.zeros(shape=(len(data),))
        results["events"]["hyperglycemic_events"]["l2"]["events_per_week"] = dict()
        results["events"]["hyperglycemic_events"]["l2"]["events_per_week"]["values"] = np.zeros(shape=(len(data),))

        results["events"]["hypoglycemic_events"] = dict()
        results["events"]["hypoglycemic_events"]["hypo"] = dict()
        results["events"]["hypoglycemic_events"]["hypo"]["mean_duration"] = dict()
        results["events"]["hypoglycemic_events"]["hypo"]["mean_duration"]["values"] = np.zeros(shape=(len(data),))
        results["events"]["hypoglycemic_events"]["hypo"]["events_per_week"] = dict()
        results["events"]["hypoglycemic_events"]["hypo"]["events_per_week"]["values"] = np.zeros(shape=(len(data),))
        results["events"]["hypoglycemic_events"]["l1"] = dict()
        results["events"]["hypoglycemic_events"]["l1"]["mean_duration"] = dict()
        results["events"]["hypoglycemic_events"]["l1"]["mean_duration"]["values"] = np.zeros(shape=(len(data),))
        results["events"]["hypoglycemic_events"]["l1"]["events_per_week"] = dict()
        results["events"]["hypoglycemic_events"]["l1"]["events_per_week"]["values"] = np.zeros(shape=(len(data),))
        results["events"]["hypoglycemic_events"]["l2"] = dict()
        results["events"]["hypoglycemic_events"]["l2"]["mean_duration"] = dict()
        results["events"]["hypoglycemic_events"]["l2"]["mean_duration"]["values"] = np.zeros(shape=(len(data),))
        results["events"]["hypoglycemic_events"]["l2"]["events_per_week"] = dict()
        results["events"]["hypoglycemic_events"]["l2"]["events_per_week"]["values"] = np.zeros(shape=(len(data),))

        results["events"]["extended_hypoglycemic_events"] = dict()
        results["events"]["extended_hypoglycemic_events"]["mean_duration"] = dict()
        results["events"]["extended_hypoglycemic_events"]["mean_duration"]["values"] = np.zeros(shape=(len(data),))
        results["events"]["extended_hypoglycemic_events"]["events_per_week"] = dict()
        results["events"]["extended_hypoglycemic_events"]["events_per_week"]["values"] = np.zeros(shape=(len(data),))

        for d in range(len(data)):
            r = find_hyperglycemic_events_by_level(data[d], glycemic_target=self.glycemic_target)
            results["events"]["hyperglycemic_events"]["hyper"]["mean_duration"]["values"][d] = r["hyper"]["mean_duration"]
            results["events"]["hyperglycemic_events"]["hyper"]["events_per_week"]["values"][d] = r["hyper"]["events_per_week"]
            results["events"]["hyperglycemic_events"]["l1"]["mean_duration"]["values"][d] = r["l1"]["mean_duration"]
            results["events"]["hyperglycemic_events"]["l1"]["events_per_week"]["values"][d] = r["l1"]["events_per_week"]
            results["events"]["hyperglycemic_events"]["l2"]["mean_duration"]["values"][d] = r["l2"]["mean_duration"]
            results["events"]["hyperglycemic_events"]["l2"]["events_per_week"]["values"][d] = r["l2"]["events_per_week"]

            r = find_hypoglycemic_events_by_level(data[d], glycemic_target=self.glycemic_target)
            results["events"]["hypoglycemic_events"]["hypo"]["mean_duration"]["values"][d] = r["hypo"]["mean_duration"]
            results["events"]["hypoglycemic_events"]["hypo"]["events_per_week"]["values"][d] = r["hypo"]["events_per_week"]
            results["events"]["hypoglycemic_events"]["l1"]["mean_duration"]["values"][d] = r["l1"]["mean_duration"]
            results["events"]["hypoglycemic_events"]["l1"]["events_per_week"]["values"][d] = r["l1"]["events_per_week"]
            results["events"]["hypoglycemic_events"]["l2"]["mean_duration"]["values"][d] = r["l2"]["mean_duration"]
            results["events"]["hypoglycemic_events"]["l2"]["events_per_week"]["values"][d] = r["l2"]["events_per_week"]

            r = find_extended_hypoglycemic_events(data[d])
            results["events"]["extended_hypoglycemic_events"]["mean_duration"]["values"][d] = r["mean_duration"]
            results["events"]["extended_hypoglycemic_events"]["events_per_week"]["values"][d] = r["events_per_week"]

        stat = ["mean", "std", "median"]
        f_stat = [np.nanmean, np.nanstd, np.nanmedian]
        prc = ["prc_5","prc_25","prc_75","prc_95"]
        qs = [5, 25, 75, 95]
        for s in range(len(stat)):
            results["events"]["hyperglycemic_events"]["hyper"]["mean_duration"][stat[s]] = f_stat[s](results["events"]["hyperglycemic_events"]["hyper"]["mean_duration"]["values"])
            results["events"]["hyperglycemic_events"]["hyper"]["events_per_week"][stat[s]] = f_stat[s](results["events"]["hyperglycemic_events"]["hyper"]["events_per_week"]["values"])
            results["events"]["hyperglycemic_events"]["l1"]["mean_duration"][stat[s]] = f_stat[s](results["events"]["hyperglycemic_events"]["l1"]["mean_duration"]["values"])
            results["events"]["hyperglycemic_events"]["l1"]["events_per_week"][stat[s]] = f_stat[s](results["events"]["hyperglycemic_events"]["l1"]["events_per_week"]["values"])
            results["events"]["hyperglycemic_events"]["l2"]["mean_duration"][stat[s]] = f_stat[s](results["events"]["hyperglycemic_events"]["l2"]["mean_duration"]["values"])
            results["events"]["hyperglycemic_events"]["l2"]["events_per_week"][stat[s]] = f_stat[s](results["events"]["hyperglycemic_events"]["l2"]["events_per_week"]["values"])

            results["events"]["hypoglycemic_events"]["hypo"]["mean_duration"][stat[s]] = f_stat[s](results["events"]["hypoglycemic_events"]["hypo"]["mean_duration"]["values"])
            results["events"]["hypoglycemic_events"]["hypo"]["events_per_week"][stat[s]] = f_stat[s](results["events"]["hypoglycemic_events"]["hypo"]["events_per_week"]["values"])
            results["events"]["hypoglycemic_events"]["l1"]["mean_duration"][stat[s]] = f_stat[s](results["events"]["hypoglycemic_events"]["l1"]["mean_duration"]["values"])
            results["events"]["hypoglycemic_events"]["l1"]["events_per_week"][stat[s]] = f_stat[s](results["events"]["hypoglycemic_events"]["l1"]["events_per_week"]["values"])
            results["events"]["hypoglycemic_events"]["l2"]["mean_duration"][stat[s]] = f_stat[s](results["events"]["hypoglycemic_events"]["l2"]["mean_duration"]["values"])
            results["events"]["hypoglycemic_events"]["l2"]["events_per_week"][stat[s]] = f_stat[s](results["events"]["hypoglycemic_events"]["l2"]["events_per_week"]["values"])

            results["events"]["extended_hypoglycemic_events"]["mean_duration"][stat[s]] = f_stat[s](results["events"]["extended_hypoglycemic_events"]["mean_duration"]["values"])
            results["events"]["extended_hypoglycemic_events"]["events_per_week"][stat[s]] = f_stat[s](results["events"]["extended_hypoglycemic_events"]["events_per_week"]["values"])

        for s in range(len(prc)):
            results["events"]["hyperglycemic_events"]["hyper"]["mean_duration"][prc[s]] = np.nanpercentile(results["events"]["hyperglycemic_events"]["hyper"]["mean_duration"]["values"], qs[s])
            results["events"]["hyperglycemic_events"]["hyper"]["events_per_week"][prc[s]] = np.nanpercentile(results["events"]["hyperglycemic_events"]["hyper"]["events_per_week"]["values"], qs[s])
            results["events"]["hyperglycemic_events"]["l1"]["mean_duration"][prc[s]] = np.nanpercentile(results["events"]["hyperglycemic_events"]["l1"]["mean_duration"]["values"], qs[s])
            results["events"]["hyperglycemic_events"]["l1"]["events_per_week"][prc[s]] = np.nanpercentile(results["events"]["hyperglycemic_events"]["l1"]["events_per_week"]["values"], qs[s])
            results["events"]["hyperglycemic_events"]["l2"]["mean_duration"][prc[s]] = np.nanpercentile(results["events"]["hyperglycemic_events"]["l2"]["mean_duration"]["values"], qs[s])
            results["events"]["hyperglycemic_events"]["l2"]["events_per_week"][prc[s]] = np.nanpercentile(results["events"]["hyperglycemic_events"]["l2"]["events_per_week"]["values"], qs[s])

            results["events"]["hypoglycemic_events"]["hypo"]["mean_duration"][prc[s]] = np.nanpercentile(results["events"]["hypoglycemic_events"]["hypo"]["mean_duration"]["values"], qs[s])
            results["events"]["hypoglycemic_events"]["hypo"]["events_per_week"][prc[s]] = np.nanpercentile(results["events"]["hypoglycemic_events"]["hypo"]["events_per_week"]["values"], qs[s])
            results["events"]["hypoglycemic_events"]["l1"]["mean_duration"][prc[s]] = np.nanpercentile(results["events"]["hypoglycemic_events"]["l1"]["mean_duration"]["values"], qs[s])
            results["events"]["hypoglycemic_events"]["l1"]["events_per_week"][prc[s]] = np.nanpercentile(results["events"]["hypoglycemic_events"]["l1"]["events_per_week"]["values"], qs[s])
            results["events"]["hypoglycemic_events"]["l2"]["mean_duration"][prc[s]] = np.nanpercentile(results["events"]["hypoglycemic_events"]["l2"]["mean_duration"]["values"], qs[s])
            results["events"]["hypoglycemic_events"]["l2"]["events_per_week"][prc[s]] = np.nanpercentile(results["events"]["hypoglycemic_events"]["l2"]["events_per_week"]["values"], qs[s])

            results["events"]["extended_hypoglycemic_events"]["mean_duration"][prc[s]] = np.nanpercentile(results["events"]["extended_hypoglycemic_events"]["mean_duration"]["values"], qs[s])
            results["events"]["extended_hypoglycemic_events"]["events_per_week"][prc[s]] = np.nanpercentile(results["events"]["extended_hypoglycemic_events"]["events_per_week"]["values"], qs[s])

        return results

    def compare_two_arms(self, arm_1, arm_2, is_paired, alpha):
        """
        Analyzes and compares glucose data of one arm.

        Parameters
        ----------
        arm_1: list of pd.DataFrame
            List of pandas dataframes with a column `glucose` containing the glucose data
            to analyze (in mg/dl). These are the data of the first arm
        arm_2: list of pd.DataFrame
            List of pandas dataframes with a column `glucose` containing the glucose data
            to analyze (in mg/dl). These are the data of the second arm
         is_paired: bool
            A boolean flag defining whether to run paired or unpaired analysis. Commonly paired tests are performed
            when data of the same patients are present in both arms, unpaired otherwise
        alpha: float
            The significance level to use during the statistical analysis

        Returns
        -------
        results: dict
            A dictionary containing two dictionaries (`arm_1` and `arm_2`) each containing the results of the analysis
             of the two arms with fields:
            - variability: dict
                A dictionary containing the values of the variability related metrics.
            - time_in_ranges: dict
                A dictionary containing the values of the time in range related metrics.
            - risk: dict
                A dictionary containing the values of the risk related metrics.
            - glycemic_transformation: dict
                A dictionary containing the values of the glycemic_transformation related metrics.
            - data_quality: dict
                A dictionary containing the values of the data quality related metrics.
            - events: dict
                A dictionary containing the values of the events related metrics.

         stats: dict
            A structure that contains for each of the considered metrics the result of the statistical test with
            field `p` (p-value value) and `h` null hypothesis accepted or rejected. Statistical tests are:
                - t-test if the test `is_paired` and the samples are both gaussian distributed
                (checked with the Lilliefors test)
                - unpaired t-test if the test not `is_paired` and the samples are both gaussian distributed
                (checked with the Lilliefors test)
                - Wilcoxon rank test if the test `is_paired` and at least one of the samples is not gaussian distributed
                (checked with the Lilliefors test)
                - Mann-Whitney U-test if the test not `is_paired` and at least one of the samples is not gaussian
                distributed (checked with the Lilliefors test).
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
        for d in arm_1:
            # Check input
            check_dataframe(d)
            check_data_columns(d)
            check_homogeneous_timegrid(d)
        for d in arm_2:
            # Check input
            check_dataframe(d)
            check_data_columns(d)
            check_homogeneous_timegrid(d)

        results = dict()
        results["arm_1"] = self.analyze_one_arm(arm_1)
        results["arm_2"] = self.analyze_one_arm(arm_2)
        stats = dict()

        # Variability
        stats["variability"] = dict()
        metric_list_name = ['mean_glucose', 'median_glucose', 'std_glucose', 'cv_glucose', 'range_glucose',
                            'iqr_glucose',
                            'auc_glucose', 'gmi', 'cogi', 'conga', 'j_index', 'mage_index', 'mage_minus_index',
                            'mage_plus_index',
                            'ef_index', 'modd', 'sddm_index', 'sdw_index', 'std_glucose_roc', 'cvga']

        for m in metric_list_name:
            r1 = results["arm_1"]["variability"][m]["values"]
            r2 = results["arm_2"]["variability"][m]["values"]

            # Correct for lilliefors behaviour
            if(r1[~np.isnan(r1)].size >= 4):
                ks, p1 = lilliefors(r1[~np.isnan(r1)])
            else:
                p1 = 0
            if (r2[~np.isnan(r2)].size >= 4):
                ks, p2 = lilliefors(r2[~np.isnan(r2)])
            else:
                p2 = 0

            stats["variability"][m] = dict()
            if np.sum(~np.isnan(r1)) < 4 or np.sum(~np.isnan(r2)) < 4 or ((p1 > 0.05 or np.isnan(p1)) and (p2 > 0.05 or np.isnan(p2))):
                t = ttest_ind(r1, r2, nan_policy="omit")
                stats["variability"][m]["p"] = t.pvalue
                if np.isnan(stats["variability"][m]["p"]):
                    stats["variability"][m]["h"] = np.nan
                else:
                    stats["variability"][m]["h"] = 1 * (t.pvalue < alpha)
            else:
                if is_paired:
                    idxs = np.where(np.logical_and(~np.isnan(r1), ~np.isnan(r2)))[0]
                    if np.all(r1[idxs] - r2[idxs]) == 0:
                        stats["variability"][m]["h"] = 0
                        stats["variability"][m]["p"] = 1
                    else:
                        t = wilcoxon(r1, r2, nan_policy="omit")
                        stats["variability"][m]["p"] = t.pvalue
                        if np.isnan(stats["variability"][m]["p"]):
                            stats["variability"][m]["h"] = np.nan
                        else:
                            stats["variability"][m]["h"] = 1 * (t.pvalue < alpha)
                else:
                    t = mannwhitneyu(r1, r2)
                    stats["variability"][m]["p"] = t.pvalue
                    if np.isnan(stats["variability"][m]["p"]):
                        stats["variability"][m]["h"] = np.nan
                    else:
                        stats["variability"][m]["h"] = 1 * (t.pvalue < alpha)

        # Time in ranges
        stats["time_in_ranges"] = dict()
        metric_list_name = ['time_in_target', 'time_in_tight_target', 'time_in_hypoglycemia',
                            'time_in_l1_hypoglycemia', 'time_in_l2_hypoglycemia', 'time_in_hyperglycemia',
                            'time_in_l1_hyperglycemia', 'time_in_l2_hyperglycemia']

        for m in metric_list_name:
            r1 = results["arm_1"]["time_in_ranges"][m]["values"]
            r2 = results["arm_2"]["time_in_ranges"][m]["values"]

            # Correct for lilliefors behaviour
            if (r1[~np.isnan(r1)].size >= 4):
                ks, p1 = lilliefors(r1[~np.isnan(r1)])
            else:
                p1 = 0
            if (r2[~np.isnan(r2)].size >= 4):
                ks, p2 = lilliefors(r2[~np.isnan(r2)])
            else:
                p2 = 0

            stats["time_in_ranges"][m] = dict()
            if np.sum(~np.isnan(r1)) < 4 or np.sum(~np.isnan(r2)) < 4 or (
                    (p1 > 0.05 or np.isnan(p1)) and (p2 > 0.05 or np.isnan(p2))):
                t = ttest_ind(r1, r2, nan_policy="omit")
                stats["time_in_ranges"][m]["p"] = t.pvalue
                if np.isnan(stats["time_in_ranges"][m]["p"]):
                    stats["time_in_ranges"][m]["h"] = np.nan
                else:
                    stats["time_in_ranges"][m]["h"] = 1 * (t.pvalue < alpha)
            else:
                if is_paired:
                    idxs = np.where(np.logical_and(~np.isnan(r1), ~np.isnan(r2)))[0]
                    if np.all(r1[idxs] - r2[idxs]) == 0:
                        stats["time_in_ranges"][m]["h"] = 0
                        stats["time_in_ranges"][m]["p"] = 1
                    else:
                        t = wilcoxon(r1, r2, nan_policy="omit")
                        stats["time_in_ranges"][m]["p"] = t.pvalue
                        if np.isnan(stats["time_in_ranges"][m]["p"]):
                            stats["time_in_ranges"][m]["h"] = np.nan
                        else:
                            stats["time_in_ranges"][m]["h"] = 1 * (t.pvalue < alpha)
                else:
                    t = mannwhitneyu(r1, r2)
                    stats["time_in_ranges"][m]["p"] = t.pvalue
                    if np.isnan(stats["time_in_ranges"][m]["p"]):
                        stats["time_in_ranges"][m]["h"] = np.nan
                    else:
                        stats["time_in_ranges"][m]["h"] = 1 * (t.pvalue < alpha)
        # Risk
        stats["risk"] = dict()
        metric_list_name = ['adrr', 'lbgi', 'hbgi', 'bgri', 'gri']

        for m in metric_list_name:
            r1 = results["arm_1"]["risk"][m]["values"]
            r2 = results["arm_2"]["risk"][m]["values"]

            # Correct for lilliefors behaviour
            if (r1[~np.isnan(r1)].size >= 4):
                ks, p1 = lilliefors(r1[~np.isnan(r1)])
            else:
                p1 = 0
            if (r2[~np.isnan(r2)].size >= 4):
                ks, p2 = lilliefors(r2[~np.isnan(r2)])
            else:
                p2 = 0

            stats["risk"][m] = dict()
            if np.sum(~np.isnan(r1)) < 4 or np.sum(~np.isnan(r2)) < 4 or (
                    (p1 > 0.05 or np.isnan(p1)) and (p2 > 0.05 or np.isnan(p2))):
                t = ttest_ind(r1, r2, nan_policy="omit")
                stats["risk"][m]["p"] = t.pvalue
                if np.isnan(stats["risk"][m]["p"]):
                    stats["risk"][m]["h"] = np.nan
                else:
                    stats["risk"][m]["h"] = 1 * (t.pvalue < alpha)
            else:
                if is_paired:
                    idxs = np.where(np.logical_and(~np.isnan(r1), ~np.isnan(r2)))[0]
                    if np.all(r1[idxs] - r2[idxs]) == 0:
                        stats["risk"][m]["h"] = 0
                        stats["risk"][m]["p"] = 1
                    else:
                        t = wilcoxon(r1, r2, nan_policy="omit")
                        stats["risk"][m]["p"] = t.pvalue
                        if np.isnan(stats["risk"][m]["p"]):
                            stats["risk"][m]["h"] = np.nan
                        else:
                            stats["risk"][m]["h"] = 1 * (t.pvalue < alpha)
                else:
                    t = mannwhitneyu(r1, r2)
                    stats["risk"][m]["p"] = t.pvalue
                    if np.isnan(stats["risk"][m]["p"]):
                        stats["risk"][m]["h"] = np.nan
                    else:
                        stats["risk"][m]["h"] = 1 * (t.pvalue < alpha)

        # Glycemic transformation
        stats["glycemic_transformation"] = dict()
        metric_list_name = ['grade_score', 'grade_hypo_score', 'grade_hyper_score',
                                        'grade_eu_score', 'igc', 'hypo_index', 'hyper_index',
                                        'mr_index']

        for m in metric_list_name:
            r1 = results["arm_1"]["glycemic_transformation"][m]["values"]
            r2 = results["arm_2"]["glycemic_transformation"][m]["values"]

            # Correct for lilliefors behaviour
            if (r1[~np.isnan(r1)].size >= 4):
                ks, p1 = lilliefors(r1[~np.isnan(r1)])
            else:
                p1 = 0
            if (r2[~np.isnan(r2)].size >= 4):
                ks, p2 = lilliefors(r2[~np.isnan(r2)])
            else:
                p2 = 0

            stats["glycemic_transformation"][m] = dict()
            if np.sum(~np.isnan(r1)) < 4 or np.sum(~np.isnan(r2)) < 4 or (
                    (p1 > 0.05 or np.isnan(p1)) and (p2 > 0.05 or np.isnan(p2))):
                t = ttest_ind(r1, r2, nan_policy="omit")
                stats["glycemic_transformation"][m]["p"] = t.pvalue
                if np.isnan(stats["glycemic_transformation"][m]["p"]):
                    stats["glycemic_transformation"][m]["h"] = np.nan
                else:
                    stats["glycemic_transformation"][m]["h"] = 1 * (t.pvalue < alpha)
            else:
                if is_paired:
                    idxs = np.where(np.logical_and(~np.isnan(r1), ~np.isnan(r2)))[0]
                    if np.all(r1[idxs] - r2[idxs]) == 0:
                        stats["glycemic_transformation"][m]["h"] = 0
                        stats["glycemic_transformation"][m]["p"] = 1
                    else:
                        t = wilcoxon(r1, r2, nan_policy="omit")
                        stats["glycemic_transformation"][m]["p"] = t.pvalue
                        if np.isnan(stats["glycemic_transformation"][m]["p"]):
                            stats["glycemic_transformation"][m]["h"] = np.nan
                        else:
                            stats["glycemic_transformation"][m]["h"] = 1 * (t.pvalue < alpha)
                else:
                    t = mannwhitneyu(r1, r2)
                    stats["glycemic_transformation"][m]["p"] = t.pvalue
                    if np.isnan(stats["glycemic_transformation"][m]["p"]):
                        stats["glycemic_transformation"][m]["h"] = np.nan
                    else:
                        stats["glycemic_transformation"][m]["h"] = 1 * (t.pvalue < alpha)

        # Data quality
        stats["data_quality"] = dict()
        metric_list_name = ['number_days_of_observation','missing_glucose_percentage']

        for m in metric_list_name:
            r1 = results["arm_1"]["data_quality"][m]["values"]
            r2 = results["arm_2"]["data_quality"][m]["values"]

            # Correct for lilliefors behaviour
            if (r1[~np.isnan(r1)].size >= 4):
                ks, p1 = lilliefors(r1[~np.isnan(r1)])
            else:
                p1 = 0
            if (r2[~np.isnan(r2)].size >= 4):
                ks, p2 = lilliefors(r2[~np.isnan(r2)])
            else:
                p2 = 0

            stats["data_quality"][m] = dict()
            if np.sum(~np.isnan(r1)) < 4 or np.sum(~np.isnan(r2)) < 4 or (
                    (p1 > 0.05 or np.isnan(p1)) and (p2 > 0.05 or np.isnan(p2))):
                t = ttest_ind(r1, r2, nan_policy="omit")
                stats["data_quality"][m]["p"] = t.pvalue
                if np.isnan(stats["data_quality"][m]["p"]):
                    stats["data_quality"][m]["h"] = np.nan
                else:
                    stats["data_quality"][m]["h"] = 1 * (t.pvalue < alpha)
            else:
                if is_paired:
                    idxs = np.where(np.logical_and(~np.isnan(r1), ~np.isnan(r2)))[0]
                    if np.all(r1[idxs] - r2[idxs]) == 0:
                        stats["data_quality"][m]["h"] = 0
                        stats["data_quality"][m]["p"] = 1
                    else:
                        t = wilcoxon(r1, r2, nan_policy="omit")
                        stats["data_quality"][m]["p"] = t.pvalue
                        if np.isnan(stats["data_quality"][m]["p"]):
                            stats["data_quality"][m]["h"] = np.nan
                        else:
                            stats["data_quality"][m]["h"] = 1 * (t.pvalue < alpha)
                else:
                    t = mannwhitneyu(r1, r2)
                    stats["data_quality"][m]["p"] = t.pvalue
                    if np.isnan(stats["data_quality"][m]["p"]):
                        stats["data_quality"][m]["h"] = np.nan
                    else:
                        stats["data_quality"][m]["h"] = 1 * (t.pvalue < alpha)

        # Events
        stats["events"] = dict()
        stats["events"]["hypoglycemic_events"] = dict()
        stats["events"]["hypoglycemic_events"]["hypo"] = dict()
        stats["events"]["hypoglycemic_events"]["l1"] = dict()
        stats["events"]["hypoglycemic_events"]["l2"] = dict()
        stats["events"]["hyperglycemic_events"] = dict()
        stats["events"]["hyperglycemic_events"]["hyper"] = dict()
        stats["events"]["hyperglycemic_events"]["l1"] = dict()
        stats["events"]["hyperglycemic_events"]["l2"] = dict()

        cat = ["hypoglycemic_events"]
        sub_cat = ['hypo', 'l1', 'l2']
        metric_list_name = ['mean_duration', 'events_per_week']

        for c in cat:
            for s in sub_cat:
                for m in metric_list_name:
                    r1 = results["arm_1"]["events"][c][s][m]["values"]
                    r2 = results["arm_2"]["events"][c][s][m]["values"]
                    # Correct for lilliefors behaviour
                    if (r1[~np.isnan(r1)].size >= 4):
                        ks, p1 = lilliefors(r1[~np.isnan(r1)])
                    else:
                        p1 = 0
                    if (r2[~np.isnan(r2)].size >= 4):
                        ks, p2 = lilliefors(r2[~np.isnan(r2)])
                    else:
                        p2 = 0

                    stats["events"][c][s][m] = dict()
                    if np.sum(~np.isnan(r1)) < 4 or np.sum(~np.isnan(r2)) < 4 or ((p1 > 0.05 or np.isnan(p1)) and (p2 > 0.05 or np.isnan(p2))):
                        t = ttest_ind(r1, r2, nan_policy="omit")
                        stats["events"][c][s][m]["p"] = t.pvalue
                        if np.isnan(stats["events"][c][s][m]["p"]):
                            stats["events"][c][s][m]["h"] = np.nan
                        else:
                            stats["events"][c][s][m]["h"] = 1 * (t.pvalue < alpha)
                    else:
                        if is_paired:
                            idxs = np.where(np.logical_and(~np.isnan(r1), ~np.isnan(r2)))[0]
                            if np.all(r1[idxs] - r2[idxs]) == 0:
                                stats["events"][c][s][m]["h"] = np.nan
                                stats["events"][c][s][m]["p"] = np.nan
                            else:
                                t = wilcoxon(r1, r2, nan_policy="omit")
                                stats["events"][c][s][m]["p"] = t.pvalue
                                if np.isnan(stats["events"][c][s][m]["p"]):
                                    stats["events"][c][s][m]["h"] = np.nan
                                else:
                                    stats["events"][c][s][m]["h"] = 1 * (t.pvalue < alpha)
                        else:
                            t = mannwhitneyu(r1, r2)
                            stats["events"][c][s][m]["p"] = t.pvalue
                            if np.isnan(stats["events"][c][s][m]["p"]):
                                stats["events"][c][s][m]["h"] = np.nan
                            else:
                                stats["events"][c][s][m]["h"] = 1 * (t.pvalue < alpha)

        cat = ["hyperglycemic_events"]
        sub_cat = ['hyper', 'l1', 'l2']
        metric_list_name = ['mean_duration', 'events_per_week']

        for c in cat:
            for s in sub_cat:
                for m in metric_list_name:
                    r1 = results["arm_1"]["events"][c][s][m]["values"]
                    r2 = results["arm_2"]["events"][c][s][m]["values"]
                    # Correct for lilliefors behaviour
                    if (r1[~np.isnan(r1)].size >= 4):
                        ks, p1 = lilliefors(r1[~np.isnan(r1)])
                    else:
                        p1 = 0
                    if (r2[~np.isnan(r2)].size >= 4):
                        ks, p2 = lilliefors(r2[~np.isnan(r2)])
                    else:
                        p2 = 0

                    stats["events"][c][s][m] = dict()
                    if np.sum(~np.isnan(r1)) < 4 or np.sum(~np.isnan(r2)) < 4 or ((p1 > 0.05 or np.isnan(p1)) and (p2 > 0.05 or np.isnan(p2))):
                        t = ttest_ind(r1, r2, nan_policy="omit")
                        stats["events"][c][s][m]["p"] = t.pvalue
                        if np.isnan(stats["events"][c][s][m]["p"]):
                            stats["events"][c][s][m]["h"] = np.nan
                        else:
                            stats["events"][c][s][m]["h"] = 1 * (t.pvalue < alpha)
                    else:
                        if is_paired:
                            idxs = np.where(np.logical_and(~np.isnan(r1), ~np.isnan(r2)))[0]
                            if np.all(r1[idxs] - r2[idxs]) == 0:
                                stats["events"][c][s][m]["h"] = np.nan
                                stats["events"][c][s][m]["p"] = np.nan
                            else:
                                t = wilcoxon(r1, r2, nan_policy="omit")
                                stats["events"][c][s][m]["p"] = t.pvalue
                                if np.isnan(stats["events"][c][s][m]["p"]):
                                    stats["events"][c][s][m]["h"] = np.nan
                                else:
                                    stats["events"][c][s][m]["h"] = 1 * (t.pvalue < alpha)
                        else:
                            t = mannwhitneyu(r1, r2)
                            stats["events"][c][s][m]["p"] = t.pvalue
                            if np.isnan(stats["events"][c][s][m]["p"]):
                                stats["events"][c][s][m]["h"] = np.nan
                            else:
                                stats["events"][c][s][m]["h"] = 1 * (t.pvalue < alpha)

        stats["events"]["extended_hypoglycemic_events"] = dict()
        for m in metric_list_name:
            r1 = results["arm_1"]["events"]["extended_hypoglycemic_events"][m]["values"]
            r2 = results["arm_2"]["events"]["extended_hypoglycemic_events"][m]["values"]
            # Correct for lilliefors behaviour
            if (r1[~np.isnan(r1)].size >= 4):
                ks, p1 = lilliefors(r1[~np.isnan(r1)])
            else:
                p1 = 0
            if (r2[~np.isnan(r2)].size >= 4):
                ks, p2 = lilliefors(r2[~np.isnan(r2)])
            else:
                p2 = 0
            stats["events"]["extended_hypoglycemic_events"][m] = dict()
            if np.sum(~np.isnan(r1)) < 4 or np.sum(~np.isnan(r2)) < 4 or ((p1 > 0.05 or np.isnan(p1)) and (p2 > 0.05 or np.isnan(p2))):
                t = ttest_ind(r1, r2, nan_policy="omit")
                stats["events"]["extended_hypoglycemic_events"][m]["p"] = t.pvalue
                if np.isnan(stats["events"]["extended_hypoglycemic_events"][m]["p"]):
                    stats["events"]["extended_hypoglycemic_events"][m]["h"] = np.nan
                else:
                    stats["events"]["extended_hypoglycemic_events"][m]["h"] = 1 * (t.pvalue < alpha)
            else:
                if is_paired:
                    idxs = np.where(np.logical_and(~np.isnan(r1), ~np.isnan(r2)))[0]
                    if np.all(r1[idxs] - r2[idxs]) == 0:
                        stats["events"]["extended_hypoglycemic_events"][m]["h"] = np.nan
                        stats["events"]["extended_hypoglycemic_events"][m]["p"] = np.nan
                    else:
                        t = wilcoxon(r1, r2, nan_policy="omit")
                        stats["events"]["extended_hypoglycemic_events"][m]["p"] = t.pvalue
                        if np.isnan(stats["events"]["extended_hypoglycemic_events"][m]["p"]):
                            stats["events"]["extended_hypoglycemic_events"][m]["h"] = np.nan
                        else:
                            stats["events"]["extended_hypoglycemic_events"][m]["h"] = 1 * (t.pvalue < alpha)
                else:
                    t = mannwhitneyu(r1, r2)
                    stats["events"]["extended_hypoglycemic_events"][m]["p"] = t.pvalue
                    if np.isnan(stats["events"]["extended_hypoglycemic_events"][m]["p"]):
                        stats["events"]["extended_hypoglycemic_events"][m]["h"] = np.nan
                    else:
                        stats["events"]["extended_hypoglycemic_events"][m]["h"] = 1 * (t.pvalue < alpha)

        return results, stats
