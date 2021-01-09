class EvaluationData:
    """ A container to store the evaluation data as it is passed from the evaluate planner to the metric

    Attributes:
        problems:
            The list of problems the planner was run on.

        (Note that the following attributes have been precomputed by the Evaluation_Planner class)

        total_successes:
            The proportion of successes as a decimal between 0 and 1.
        all_times:
            A list of times, one time for each Problem solved.
        all_battery_levels:
            A list of lists of battery levels, one list for each Problem solved.
        failed_problems:
            A list of Problems the planner failed.
    """

    def __init__(self, problems, total_successes, all_times, all_battery_levels, failed_problems):
        self.problems = problems
        self.total_successes = total_successes
        self.all_times = all_times
        self.all_battery_levels = all_battery_levels
        self.failed_problems = failed_problems
