from ocean_navigation_simulator.evaluation.evaluate_planner import EvaluatePlanner


class EvaluateHighLevelPlanner(EvaluatePlanner):
    """Evaluates a planner on a set of problems

    Attributes:
        problem_set:
            A ProblemSet instance which will supply the problems.
        project_dir:
            A string giving the path to the project directory.
        wypt_contr:
            A WaypointTrackingController used to compare different planners
    """

    def __init__(self, problem_set, project_dir, wypt_contr):
        super().__init__(problem_set, project_dir)
        self.wypt_contr = wypt_contr

    def evaluate_planner(self, metrics_strings=('success_rate', 'avg_time', 'avg_bat_level', 'avg_bat_level_variance',
                                                'avg_bat_level_below_threshold'), plot=True):
        """ Evaluates the planner on all the problems in self.problem_set.

        Args:
            wypt_contr: An instance of the WaypointTrackingController class

        Returns:
            An EvaluationData instance.
        """
        # TODO: use similar logic as the evaluate_WTC method in evalute_waypoint_controller.py
        pass
