import numpy as np


class EvaluationMetric:
    """
    The protocol class for any evaluation metric.

    Attributes:
        data:
            An EvaluationData instance containing the pertinent data.
    """

    def __init__(self, evaluation_data):
        self.data = evaluation_data
        self.results = self.evaluate()

    def evaluate(self):
        """ Applies some metric on the given data """
        raise NotImplementedError()

    def plot_results(self):
        raise NotImplementedError()

    @classmethod
    def create_metric(cls, metric_string, evaluation_data):
        """Creates an instance of the appropriate metric

        Args:
            metric_string:
                See evaluate_planner in evaluation_planner.py.
            evaluation_data:
                An instance of EvaluationData containing the pertinent data for the metric.

        Returns:
            An instance of an EvaluationMetric subclass, e.g. SuccessRateMetric.
        """
        if metric_string == 'success_rate':
            return SuccessRateMetric(evaluation_data)
        elif metric_string == 'avg_time':
            return AvgTimeMetric(evaluation_data)
        elif metric_string == 'avg_bat_level':
            return AvgBatLevelMetric(evaluation_data)
        elif metric_string == 'avg_bat_level_variance':
            return AvgBatLevelVarMetric(evaluation_data)
        elif metric_string == 'avg_bat_level_below_threshold':
            return AvgBatLevelBelowThresholdMetric(evaluation_data)


class SuccessRateMetric(EvaluationMetric):

    def evaluate(self):
        """Finds the success rate, i.e. the proportion of problems successively solved.

        Returns:
            A float.
        """
        return self.data.total_successes / len(self.data.problems)

    def plot_results(self):
        print("\nPERCENT SUCCESSFUL: {} %".format(round(self.results, ndigits=5)))


class AvgTimeMetric(EvaluationMetric):

    def evaluate(self):
        """Finds the average time it takes the planner to solve a problem.

        Returns:
            A float.
        """
        return np.average(self.data.all_times)

    def plot_results(self):
        print("\nAVERAGE TIME: ", round(self.results, ndigits=5))


class AvgBatLevelMetric(EvaluationMetric):

    def evaluate(self):
        """Finds the average battery level across all states in all problems.

        Returns:
            A float.
        """
        average = np.average([np.average(battery_levels) for battery_levels in self.data.all_battery_levels])
        return average

    def plot_results(self):
        print("\nAVERAGE BATTERY LEVEL: ", round(self.results, ndigits=5))


class AvgBatLevelVarMetric(EvaluationMetric):

    def evaluate(self):
        """Finds the average battery level variance across all problems. The variance is calculated for each problem,
        and then averaged.

        Returns:
            A float.
        """
        average = np.average([np.var(battery_levels) for battery_levels in self.data.all_battery_levels])
        return average

    def plot_results(self):
        print("\nAVERAGE BATTERY VARIANCE: ", round(self.results, ndigits=5))


class AvgBatLevelBelowThresholdMetric(EvaluationMetric):

    def evaluate(self):
        """Finds the average percent of time the battery level is below a given threshold across all states in all
        problems.

         Returns:
             A float.
         """

        def percent_below(battery_levels, threshold=0.2):
            # we will create a list of all the battery levels below the threshold to calculate percent of time below
            below_thresh = list(filter(lambda bat_level: bat_level < threshold, battery_levels))
            below = len(below_thresh) / len(battery_levels)
            return below

        average = np.average([percent_below(battery_levels) for battery_levels in self.data.all_battery_levels])
        return average

    def plot_results(self):
        print("\nAVERAGE BATTERY PROPORTION BELOW THRESHOLD: {} %"
              "".format(round(self.results, ndigits=5)))