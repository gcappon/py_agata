from time_in_ranges import *
class Agata:

    def __init__(self, data, glycemic_target='diabetes'):
        self.data = data #TODO: validate data
        self.glycemic_target = glycemic_target

    def analyze_glucose_profile(self):

        results = dict()

        # Get time metrics
        results['time'] = dict()
        results['time']['time_in_target'] = time_in_target(self.data, self.glycemic_target)
        results['time']['time_in_tight_target'] = time_in_tight_target(self.data, self.glycemic_target)
        results['time']['time_in_hypoglycemia'] = time_in_hypoglycemia(self.data, self.glycemic_target)
        results['time']['time_in_l1_hypoglycemia'] = time_in_l1_hypoglycemia(self.data, self.glycemic_target)
        results['time']['time_in_l2_hypoglycemia'] = time_in_l2_hypoglycemia(self.data, self.glycemic_target)
        results['time']['time_in_hyperglycemia'] = time_in_hyperglycemia(self.data, self.glycemic_target)
        results['time']['time_in_l1_hyperglycemia'] = time_in_l1_hyperglycemia(self.data, self.glycemic_target)
        results['time']['time_in_l2_hyperglycemia'] = time_in_l2_hyperglycemia(self.data, self.glycemic_target)

        # Return results
        return results
