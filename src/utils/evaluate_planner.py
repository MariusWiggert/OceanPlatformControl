from src.utils.evaluation_data import EvaluationData
from src.utils.metrics import EvaluationMetric
from src.utils.simulator import Simulator


class EvaluatePlanner:
    """Evaluates a planner on a set of problems

    Attributes:
        problem_set:
            A ProblemSet instance which will supply the problems.
        project_dir:
            A string giving the path to the project directory.
        total_successes:
            The proportion of successes as a decimal between 0 and 1.
        all_times:
            A list of times, one time for each Problem solved.
        all_battery_levels:
            A list of lists of battery levels, one list for each Problem solved.
        failed_problems:
            A list of Problems the planner failed.
        metrics:
            A dictionary mapping metric name to metric instance. See evaluate_planner for more clarity.
    """

    def __init__(self, planner, problem_set, project_dir):
        self.planner = planner
        self.problem_set = problem_set
        self.project_dir = project_dir
        self.total_successes = 0
        self.all_times = []
        self.all_battery_levels = []
        self.failed_problems = []
        self.metrics = {}

    def evaluate_planner(self, metrics_strings=('success_rate', 'avg_time', 'avg_bat_level', 'avg_bat_level_variance',
                                                'avg_bat_level_below_threshold'), plot=True):
        """ Evaluates the planner on all the problems in self.problem_set.

        Calls self.evaluate_on_problem(...) for each problem in the problem_set. After looping through all the problems,
        populates total_successes, all_times, all_battery_levels, and failed_problems with the pertinent data.

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

        # Step 1: Run the planner on the given problem
        self.total_successes, self.all_times, self.all_battery_levels, self.failed_problems = 0, [], [], []
        for problem in self.problem_set.problems:
            success, time, battery_levels = self.evaluate_on_problem(problem=problem)
            self.total_successes += success
            if success:
                self.all_times.append(time)
                self.all_battery_levels.append(battery_levels)
            else:
                self.failed_problems.append(problem)

        # Step 2: Package the data for the metrics
        evaluation_data = EvaluationData(problems=self.problem_set.problems,
                                         all_battery_levels=self.all_battery_levels,
                                         total_successes=self.total_successes,
                                         all_times=self.all_times,
                                         failed_problems=self.failed_problems)

        # Step 3: Set up the metrics
        metrics = []
        for metric_string in metrics_strings:
            metric = EvaluationMetric.create_metric(metric_string=metric_string, evaluation_data=evaluation_data)
            if metric:
                metrics.append(metric)
                self.metrics[metric_string] = metric

        # Step 4: Evaluate the metrics
        for metric in metrics:
            if plot:
                metric.plot_results()

    def evaluate_on_problem(self, problem, sim_config='simulator.yaml'):
        """Evaluate the planner on the given problem by some metrics.

        Creates and runs a Simulator for the given planner and problem. Extracts information from the simulator's
        trajectory.

        Args:
            problem:
                A Problem
            sim_config:
                A YAML file for the Simulator configurations.

        Returns:
            Currently returns (success, time, list of battery_levels), but this will most likely be added to over time.
        """
        # Step 1: Set the planner's problem to the given problem
        self.planner.problem = problem

        # Step 2: Create and run the simulator
        sim = Simulator(planner=self.planner, problem=problem, project_dir=self.project_dir, sim_config=sim_config)
        success = sim.run()
        if not success:
            return False, None, (None, None, None)

        # Step 3: extract the "time" variable of the final state, which is the last element of the last list
        time = sim.trajectory[-1][-1]

        # Step 4: extract a list of all the battery levels
        battery_levels = sim.trajectory[2]

        return success, time, battery_levels
