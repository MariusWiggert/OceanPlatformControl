# from ocean_navigation_simulator.planners import Planner
from ocean_navigation_simulator.utils import simulation_utils
import math
import casadi as ca
import numpy as np


class AStarState:
    """A State in A* and minimumThrustController

    Attributes:
        lon: the longitude
        lat: the latitude
        time_to: a dictionary mapping each state to the time to get to the state
        bat_level: the current battery level of the state
        heading: the heading used to GET to this state
        thrust: the thrust applied to GET to this state
        adjacent: a list of all points (lon, lat), that are adjacent to this point

    """
    u_curr_func, v_curr_func = None, None
    dlon, dlat = None, None
    time_to = None
    problem = None
    heuristic = None
    in_bounds = None
    discretization = None
    offsets = []
    matrices = {}

    def __init__(self, lon, lat, bat_level, heading=None, thrust=None):
        self.lon = lon
        self.lat = lat
        self.bat_level = bat_level
        self.heading = heading
        self.thrust = thrust
        self.adjacent = self.adjacent_points()

    def adjacent_points(self):
        """ Returns a list of all valid points that are adjacent in the 8 directions """
        dlon, dlat = self.dlon, self.dlat
        assert dlon is not None and dlat is not None, "need to set dlon and dlat first"
        for lon_offset, lat_offset in AStarState.offsets:
            point = (self.lon + lon_offset, self.lat + lat_offset)
            if AStarState.in_bounds.in_ocean(point):
                yield point

    @classmethod
    def fill_matrices(cls):
        for dlon, dlat in AStarState.offsets:
            parallel_basis_vector = cls.normalize(np.array([dlon, dlat]))
            perp_basis_vector = cls.normalize(np.array([dlat, -dlon]))
            matrix = np.array([parallel_basis_vector, perp_basis_vector])
            cls.matrices[round(dlon, 4), round(dlat, 4)] = matrix

    def neighbors(self):
        """ Returns all reachable neighboring states in A, along with the time to get there

        Returns:
            time, next state
        """
        # iterate over all valid adjacent points
        for lon, lat in self.adjacent:
            for thrust, heading, time in self.actuate_towards(lon, lat):
                # check if it exceeds the available energy from the battery
                bat_level = self.change_bat_level(thrust, time)
                if bat_level >= 0.1:
                    yield time, AStarState(lon=lon, lat=lat, bat_level=bat_level, heading=heading, thrust=thrust)

    def __lt__(self, other):
        return self.time_to[self] + AStarState.heuristic(self) < self.time_to[other] + AStarState.heuristic(other)

    def __hash__(self):
        return hash((self.lon, self.lat, self.bat_level))

    def __repr__(self):
        return "State(lon: {0}, lat: {1}, battery_level: {2})".format(self.lon, self.lat, self.bat_level)

    def __eq__(self, other):
        if abs(self.lon - other.lon) > 0.01 or abs(self.lat - other.lat) > 0.01:
            return False

        if self.bat_level is None or other.bat_level is None:
            return True
        else:
            return self.bat_level_round(other.bat_level) == self.bat_level_round(self.bat_level)

    def bat_level_round(self, bat_level):
        return math.floor(bat_level * 20) / 20

    @staticmethod
    def normalize(v):
        return v / math.sqrt(np.sum(v ** 2))

    @classmethod
    def set_dlon_dlat(cls, discretization, lon, lat, target_lon, target_lat):
        """ Sets the dlon and dlat variables of the class appropriately (thereby creating a grid).

        Args:
            lon: longitude, a float
            lat: latitude, a float
            target_lon: the target longitude, a float
            target_lat: the target latitude, a float
        Returns:
            min_thrust, heading, time
        """
        starting_dlon, starting_dlat = discretization, discretization
        lon_dist, lat_dist = abs(target_lon - lon), abs(target_lat - lat)

        lon_intervals, lon_extra = lon_dist // starting_dlon, lon_dist % starting_dlon
        lat_intervals, lat_extra = lat_dist // starting_dlat, lat_dist % starting_dlat

        if lon_intervals:
            cls.dlon = starting_dlon + lon_extra / lon_intervals
        else:
            cls.dlon = lon_dist

        if lat_intervals:
            cls.dlat = starting_dlat + lat_extra / lat_intervals
        else:
            cls.dlat = lat_dist

        dlat, dlon = cls.dlat, cls.dlon
        AStarState.offsets = [(0, dlat), (dlon, 0), (dlon, dlat), (0, -dlat), (-dlon, 0),
                              (-dlon, -dlat), (dlon, -dlat), (-dlon, dlat)]

    def actuate_towards(self, lon, lat, v_min=0.1, distance=None, print_output=False, use_middle=False, full_send=False,
                        cushion=0):
        """ Yields thrusts

        Args:
            lon: longitude, a float
            lat: latitude, a float
            matrix: precomputed matrix containing basis vectors in direction of motion
            v_min: minimum velocity needed
            distance: precompted distance between the two points
            max_thrust: upper bound on the maximum thrust to determine feasibility
        Returns:
            min_thrust, heading, time
        """

        max_thrust = self.problem.dyn_dict['u_max'] + cushion

        # Step 1:  Find the distance between the two points
        dlon = lon - self.lon
        dlat = lat - self.lat
        distance = math.sqrt(dlon * dlon + dlat * dlat)

        if not AStarState.matrices:
            AStarState.fill_matrices()

        if use_middle:
            # Step 2: Find the current velocity vector (in middle of distance),
            # w.r.t basis with parallel vector (direction of motion) and perpendicular vector
            middle_lon = self.lon + dlon / 2
            middle_lat = self.lat + dlat / 2

            # TODO: currently doesn't take in the time, should address that
            u_curr = self.u_curr_func(ca.vertcat(middle_lat, middle_lon))
            v_curr = self.v_curr_func(ca.vertcat(middle_lat, middle_lon))
        else:
            u_curr = self.u_curr_func(ca.vertcat(self.lat, self.lon))
            v_curr = self.v_curr_func(ca.vertcat(self.lat, self.lon))

        parallel_basis_vector = AStarState.normalize(np.array([dlon, dlat]))
        perp_basis_vector = AStarState.normalize(np.array([dlat, -dlon]))
        matrix = np.array([parallel_basis_vector, perp_basis_vector])

        # TODO: check that using the matrix below actually works
        # matrix = State.matrices[round(dlon, 4), round(dlat, 4)]

        # change basis of vectors w.r.t normalized basis in direction of motion
        change_of_basis = np.linalg.inv(matrix)
        current_vector = np.matmul(np.array([u_curr, v_curr]), change_of_basis)
        curr_parallel, curr_perp = current_vector[0], current_vector[1]

        # Step 4: Set the u_perp to be the negative of the current_perp to counteract it
        u_perp = -curr_perp

        # Step 5: Set magnitude of d u_parallel = max(v_min - c_parallel, 0), heading to be direction between points.
        u_parallel_min = max(v_min - curr_parallel, 0)

        # Step 6: If u_perp is more than possible thrust, failure
        if math.sqrt(u_perp ** 2 + u_parallel_min ** 2) > max_thrust:
            if print_output:
                print("U PERP MORE THAN POSSIBLE THRUST\n Needed thrust: ",
                      math.sqrt(u_perp ** 2 + u_parallel_min ** 2), " max thrust ", max_thrust)
                print("going from {0} to {1}".format((self.lon, self.lat), (lon, lat)))
            return

        u_parallel_max = math.sqrt(max_thrust ** 2 - u_perp ** 2)

        """ If we wanted to consider many possible thrusts, too slow in practice. """
        # u_parallels = list(np.arange(u_parallel_min, u_parallel_max - 0.1, 0.1)) + [u_parallel_max]

        thrusts = [u_parallel_max if full_send else u_parallel_min]

        # for loop allows us to potentially consider more than one thrust
        for u_parallel in thrusts:
            # Step 7: Find the TIME = distance / (u_parallel + current_parallel)
            # TODO: replace hard coded constant with settings data reference
            time = distance * 111120. / (u_parallel + curr_parallel)

            # Step 8: Go back to original basis, i.e. in terms of u and v
            u_vector = np.array([u_parallel, u_perp])
            u_dir = np.matmul(u_vector, matrix)

            # Step 9: Find the thrust and heading
            raise ValueError("Not implemented due to circular import when installing with Pip..")
            # thrust_array = Planner.transform_u_dir_to_u(u_dir=u_dir)
            thrust_array = np.array([0,0])
            thrust, heading = thrust_array[0], thrust_array[1]
            yield thrust, heading, time

    def change_bat_level(self, thrust, time):
        # calculate the change in battery level when actuating a specific trust for a certain time
        bat_level_delta = self.problem.dyn_dict['charge'] - self.problem.dyn_dict['energy'] * (thrust ** 3)
        return min(self.bat_level + time * bat_level_delta, 1)

    @classmethod
    def set_fieldset(cls, problem, type='linear'):
        # setting the fieldset
        cls.u_curr_func, cls.v_curr_func = simulation_utils.get_interpolation_func(
            fieldset=problem.fieldset, type=type, fixed_time_index=problem.fixed_time_index)
