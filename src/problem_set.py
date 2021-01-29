import math
import pickle
import random

from src.problem import Problem


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
    def __init__(self, fieldset, project_dir, filename=None, num_problems=100):
        self.fieldset = fieldset
        self.project_dir = project_dir
        if filename is None:
            # random.seed(num_problems)
            self.problems = [self.create_problem() for _ in range(num_problems)]
        else:
            self.problems = self.load_problems(filename)

    def create_problem(self):
        """Randomly generates a Problem with valid x_0 and x_T.

        Iteratively produces random problems until one that fulfills the criteria in valid_start_and_end is found.

        Returns:
            A Problem.
        """
        x_0, x_T = None, None
        while not self.valid_start_and_end(x_0, x_T):
            x_0, x_T = self.random_point(), self.random_point()
        return Problem(self.fieldset, x_0, x_T, self.project_dir)

    def random_point(self):
        """Returns a random point anywhere in the grid.

        Returns:
            A point, i.e. a pair of longitude and latitude coordinates: [lon, lat].
        """
        lon = random.choice(self.fieldset.U.grid.lon)
        lat = random.choice(self.fieldset.U.grid.lat)
        return [lon, lat]

    def valid_start_and_end(self, x_0, x_T):
        """Determines whether the given start (x_0) and target (x_T) are valid.

        For a start and end to be valid, they must be sufficiently far apart, and neither point can be in the ocean.

        Args:
            x_0:
                The starting point, a pair of longitude and latitude coordinates: [lon, lat].
            x_T:
                The target point, a pair of longitude and latitude coordinates: [lon, lat].
        Returns:
            A boolean.
        """
        if x_0 is None or x_T is None:
            return False
        return self.is_far_apart(x_0, x_T) and self.in_ocean(x_0) and self.in_ocean(x_T)

    def is_far_apart(self, x_0, x_T, sep=0.5):
        """Returns whether x_0 and x_T are sufficiently far apart

        Args:
            x_0:
                The starting point, a pair of longitude and latitude coordinates: [lon, lat].
            x_T:
                The target point, a pair of longitude and latitude coordinates: [lon, lat].
            sep:
                The minimum distance between the two points.
        Returns:
            A boolean.
        """
        lon, lat, lon_target, lat_target = x_0[0], x_0[1], x_T[0], x_T[1]
        dlon = lon_target - lon
        dlat = lat_target - lat
        mag = math.sqrt(dlon * dlon + dlat * dlat)  # mag is the distance between the two points.
        return mag > sep

    def in_ocean(self, point, offset=0.1):
        """ Returns whether the point is in the ocean.

        Determines this by checking if the velocity is nonzero for this and ALL points that are "offset" distance
        about the point in the 8 directions.

        Args:
            point:
                A pair of longitude and latitude coordinates: [lon, lat].
            offset: A float which determines how far about the point to look. Increasing the value of offset will
                prevent points on the coast from being chosen.

        Returns:
            A boolean.
        """
        lon, lat = point[0], point[1]
        offsets = [(0, 0), (0, offset), (offset, 0), (offset, offset), (0, -offset),
                   (-offset, 0), (-offset, -offset), (offset, -offset), (-offset, offset)]
        for lon_offset, lat_offset in offsets:
            if self.zero_velocity(lon + lon_offset, lat + lat_offset):
                return False
        return True

    def out_of_bounds(self, coordinate, grid):
        """Determines whether the given coordinate (either lat or lon) is out of bounds for its respective grid.

        Returns:
            A boolean.
        """
        return coordinate < min(grid) or coordinate > max(grid)

    def zero_velocity(self, lon, lat):
        """Determines whether the (lon, lat) pair is zero velocity, i.e. on land.

        Returns:
            A boolean.
        """
        if self.out_of_bounds(lat, self.fieldset.U.grid.lat) or self.out_of_bounds(lon, self.fieldset.U.grid.lon):
            return False
        x = self.fieldset.U.eval(0., 0., lat, lon)
        y = self.fieldset.V.eval(0., 0., lat, lon)
        return x == 0. and y == 0.

    def load_problems(self, filename):
        """Deserializes the list of problems from the filename.

        Args:
            filename:
                A filename represented as a String, e.g. 'file.txt', that need not already exist.
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

