import pickle

from src.simulation.metrics import EvaluationMetric


class DataVisualization:
    """ Visualizes simulation data (EvaluationData instances) on selected metrics

    Attributes:
        metrics:
            A dictionary mapping metric name to metric instance. See evaluate_planner for more clarity.
    """

    def __init__(self, ):
        self.metrics = {}

    def visualize(self, filename=None, evaluation_data=None, metrics_strings=('success_rate', 'avg_time', 'avg_bat_level', 'avg_bat_level_variance',
                                                'avg_bat_level_below_threshold'), plot=True):
        """ Visualizes the evaluation_data on the metrics above

        Args:
            metrics_strings:
                A list of Metric strings which to evaluate the planner on. If no metrics are given, it simply runs the
                planner on the given problem_set. All possible strings are given below.

                Metric strings:
                - 'success_rate'
                    -> The success rate
                - 'avg_time'
                    -> The average time out of all successful results.
                - 'avg_bat_level'
                    -> The average battery level across all states in all problems)
                - 'avg_bat_level_variance'
                    -> Finds the average battery level variance across all problems. The variance is calculated for each
                    problem, and then averaged.
                - 'avg_bat_level_below_threshold'
                    -> Finds the average percent of time the battery level is below a given threshold across all states
                    in all problems.

            plot:
                A boolean indicating whether the result of the metrics should be plotted

        Returns:
            None

        """
        assert evaluation_data or filename, "Must either give a filename containing the evaluation data, or the " \
                                            "actual data "

        if filename:
            with open(filename, 'rb') as reader:
                evaluation_data = pickle.load(reader)

        metrics = []
        for metric_string in metrics_strings:
            metric = EvaluationMetric.create_metric(
                            metric_string=metric_string,
                            evaluation_data=evaluation_data)
            if metric:
                metrics.append(metric)
                self.metrics[metric_string] = metric

        # Step 4: Evaluate the metrics
        for metric in metrics:
            if plot:
                metric.plot_results()

    def visualizeMany(self, evaluation_data_list,
                  metrics_strings=('success_rate', 'avg_time', 'avg_bat_level', 'avg_bat_level_variance',
                                   'avg_bat_level_below_threshold'), plot=True):
        pass
        # TODO: Graph many EvaluationData instances, rather than just one, comparing on the same plot
