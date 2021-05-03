from src.problem_set import ProblemSet
from src.simulation.evaluation_data import EvaluationData
from src.simulation.metrics import EvaluationMetric
from src.simulation.simulator import Simulator


class EvaluatePlanner:
    """ Evaluates planner(s)/WTC(s) on a set of problems

    Attributes:
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
        filename:
            The file that the problems to be used for comparison were serialized to. The problems should be serialized
            as a list of Problems instances
    """

    def __init__(self, project_dir, filename):
        self.total_successes = []
        self.all_times = []
        self.all_battery_levels = []
        self.all_trajectories = []
        self.project_dir = project_dir
        self.all_times = []
        self.all_battery_levels = []
        self.failed_problems = []
        if filename is not None:
            self.problems = self.read_in_problems(filename=filename)

    @staticmethod
    def read_in_problems(filename):
        """ Read in the problems """
        return ProblemSet.load_problems(filename=filename)

    def evaluate_on_problem(self, planner, wypt_contr, problem, sim_config='simulator.yaml'):
        """Simluate the WTC/Planner on the given problem.

        Creates and runs a Simulator for the given planner and problem. Extracts information from the simulator's
        trajectory.

        Args:
            planner:
                The planner used for replanning and where the waypoints came from
            wypt_contr:
                The waypoint tracking controller
            problem:
                A Problem instance
            sim_config:
                A YAML file for the Simulator configurations.

        Returns:
            Currently returns (success, time, list of battery_levels), but this will most likely be added to over time.
        """
        sim = Simulator(wypt_contr=wypt_contr, planner=planner, problem=problem, project_dir=self.project_dir, sim_config=sim_config)
        try:
            success = sim.run()
        except:
            # TODO: prevent WTCs from erroring to avoid this clause
            success = False
        if not success:
            return False, None
        return success, sim.trajectory
