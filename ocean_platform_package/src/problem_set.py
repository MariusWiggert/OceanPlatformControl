import math
import pickle
import random

from src.problem import Problem, WaypointTrackingProblem
from src.utils.in_bounds_utils import InBounds


class ProblemSet:
    """Stores a list of Problems.

    If no filename is given, the list of problems is randomly created. Else, the list of problems is deserialized
    from the filename with pickle. This class is used in the EvaluatePlanner's evaluate method to provide a collection
    of Problems to test a given Planner.

    Attributes:
        fieldset:
            The fieldset contains data about ocean currents and is more rigorously defined in the
            parcels documentation.
        project_dir:
            A string giving the path to the project directory.
        problems:
            A list of Problems.
    """

    def __init__(self, fieldset, project_dir, planner=None, filename=None, num_problems=100, WTC=False):
        self.fieldset = fieldset
        self.project_dir = project_dir
        self.in_bounds = InBounds(self.fieldset)
        if filename is None:
            random.seed(num_problems)
            if WTC:
                assert planner is not None, "Must pass in a planner to create Waypoint Controller Problems "
                create = lambda: self.create_waypoint_tracking_problem(planner)
            else:
                create = self.create_problem
            self.problems = [create() for _ in range(num_problems)]
        else:
            self.problems = self.load_problems(filename)

    def create_problem(self):
        """Randomly generates a Problem with valid x_0 and x_T.

        Iteratively produces random problems until one that fulfills the criteria in valid_start_and_end is found.

        Returns:
            A Problem.
        """
        x_0, x_T = None, None
        while not self.in_bounds.valid_start_and_end(x_0, x_T):
            x_0, x_T = self.random_point(), self.random_point()
        return Problem(real_fieldset=self.fieldset, x_0_pos=x_0, x_T=x_T, project_dir=self.project_dir)

    def create_waypoint_tracking_problem(self, planner):
        """ Create and return a WaypointTrackingProblem """
        problem = self.create_problem()
        planner.run(problem)
        waypoints = planner.get_waypoints()

        return WaypointTrackingProblem.convert_problem(problem, waypoints)

    def random_point(self):
        """Returns a random point anywhere in the grid.

        Returns:
            A point, i.e. a pair of longitude and latitude coordinates: [lon, lat].
        """
        lon = random.choice(self.fieldset.U.grid.lon)
        lat = random.choice(self.fieldset.U.grid.lat)
        return [lon, lat]

    @classmethod
    def load_problems(cls, filename):
        """Deserializes the list of problems from the filename.

        Args:
            filename:
                A filename represented as a String, e.g. 'file.txt'
        Returns:
            A list of Problems.
        """
        with open(filename, 'rb') as reader:
            return pickle.load(reader)

    def save_problems(self, filename):
        """Serializes the list of problems to the filename.

        Returns:
            None
        """
        with open(filename, 'wb') as writer:
            pickle.dump(self.problems, writer)
