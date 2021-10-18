import pickle

from ocean_platform_package.src.problem import WaypointTrackingProblem
from ocean_platform_package.src.evaluation.evaluate_high_level_planner import EvaluatePlanner
from ocean_platform_package.src.evaluation.evaluation_data import EvaluationData


class EvaluateWaypointController(EvaluatePlanner):
    """ Compares and evaluates a WaypointTrackingController

    Attributes:
        planner: an instance of the Planner class, to be used to find the waypoints for the WTCs
        see rest in super class
    """

    def __init__(self, project_dir, filename, planner):
        super().__init__(project_dir=project_dir, filename=filename)
        self.planner = planner

    def compare_WTCs(self, wypt_contrs, problems_filename=None, data_filename=None, planner=None):
        """ Aggregates the performance of many TTCs with one planner on a series of problems

        Args:
            wypt_contrs:
                A list of waypoint tracking controllers
            problems_filename:
                A filename for the problems to be ran. If none, just uses self.problems
            data_filename:
                A filename for the data to be saved to. If none, doesn't save to this file
            planner:
                The planner that should be used to compare the WTCs

        Returns:
            A dictionary mapping the name of each waypoint tracking controller to the corresponding EvaluationData
            instance containing the data from the testing.
        """

        if planner is not None:
            self.planner = planner

        error = "The given file doesn't contain waypointTrackingProblems"
        assert all([type(p) == WaypointTrackingProblem for p in self.problems]), error

        all_data = {}
        # TODO: parallelize if necessary
        for wypt_contr in wypt_contrs:
            self.__init__(project_dir=self.project_dir, filename=problems_filename, planner=self.planner)
            data = self.evaluate_WTC(wypt_contr, self.planner)
            all_data[str(wypt_contr)] = data

        if data_filename is not None:
            with open(data_filename, 'wb') as writer:
                pickle.dump(all_data, writer)

        return all_data

    def evaluate_WTC(self, wypt_contr, planner):
        """ Evaluates the given WTC on all the problems saved in the class

        Args:
            wypt_contr: An instance of the WaypointTrackingController class

        Returns:
            An EvaluationData instance.
        """
        for problem in self.problems:
            wypt_contr.set_waypoints(waypoints=problem.waypoints, problem=problem)
            success, trajectory = self.evaluate_on_problem(planner=planner, wypt_contr=wypt_contr, problem=problem)
            if success:
                self.all_trajectories.append(trajectory)
            self.total_successes.append(success == 1)

        # Package the data for the metrics
        return EvaluationData(problems=self.problems,
                              total_successes=self.total_successes,
                              all_trajectories=self.all_trajectories)