class EvaluationData:
    """ A container to store the evaluation data as it is passed from the evaluate planner to the metric

    Attributes:
        problems:
            The list of problems the planner was run on.

        (Note that the following attributes have been precomputed by the Evaluation_Planner class)

        total_successes:
            A list of 0s and 1s
        all_trajectories:
            A list of trajectories the waypoint tracking controller took on each problem
    """

    def __init__(self, problems, total_successes, all_trajectories):
        self.problems = problems
        self.total_successes = total_successes
        self.all_trajectories = all_trajectories
        self.all_battery_levels = self.get_battery_levels()
        self.all_times = self.get_all_times()

    def get_battery_levels(self):
        """ Returns a list of battery levels from the evaluation_data instance

        Returns:
            A list of lists battery levels

        """
        # extract a list of all the battery levels
        return list(map(lambda traj: traj[2], self.all_trajectories))

    def get_all_times(self):
        """ Returns a list of battery levels from the evaluation_data instance

        Returns:
            A list of lists battery levels

        """
        # extract the "time" variable of the final state, which is the last element of the last list
        return list(map(lambda traj: traj[-1][-1], self.all_trajectories))