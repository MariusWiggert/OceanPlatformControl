from src.planners.planner import Planner
from src.utils import simulation_utils
import heapq
import math
import casadi as ca
import numpy as np
import pdb


class AStarPlanner(Planner):
    """Discretize the graph and run A Star

    Attributes:
        dt:
            A float giving the time, in seconds, between queries.
        see Planner class for the rest of the attributes.
    """

    def __init__(self, problem,
                 settings=None,
                 t_init=None, n=100, mode='open-loop'):
        super().__init__(problem, settings, t_init, n, mode)
        # self.dt = self.T_init / self.N
        self.dt = 10.
        self.waypoints = []
        self.control = []
        self.a_star()

    def get_next_action(self, state):
        """Grab the next action. """
        return self.control.pop(0)

    def a_star(self):
        """ Discretize the graph and find the time optimal path using A Star
        """
        # data structures
        time_to = {}
        edge_to = {}
        seen = set()
        pq = []

        # set up problem
        lon, lat = self.problem.x_0[0], self.problem.x_0[1]
        target_lon, target_lat = self.problem.x_T[0], self.problem.x_T[1]

        # set the state class variables
        State.set_dlon_dlat(lon, lat, target_lon, target_lat)
        State.set_fieldset(self.problem)

        starting_state = State(lon=lon, lat=lat, time_to=time_to, bat_level=1.0)
        pq.append(starting_state)
        time_to[starting_state] = 0
        edge_to[starting_state] = starting_state
        state = starting_state
        end_state = State(lon=target_lon, lat=target_lat, time_to=time_to, bat_level=None)

        # run Dijkstra's
        while not state == end_state:
            state = heapq.heappop(pq)
            print(state.lon, state.lat, time_to[state])
            seen.add(state)
            for time, neighbor_state in state.neighbors():
                if neighbor_state in seen:
                    continue
                if neighbor_state not in pq:
                    pq.append(neighbor_state)
                new_time = time + time_to[state]
                if neighbor_state not in time_to or new_time < time_to[neighbor_state]:
                    time_to[neighbor_state] = new_time
                    edge_to[neighbor_state] = state
            heapq.heapify(pq)

        # extract the waypoints
        while state != starting_state:
            self.waypoints.insert(0, (state.lon, state.lat))
            self.control.insert(0, (state.thrust, state.heading))
            state = edge_to[state]


class State:
    """A State in A*

    Attributes:
        lon: the longitude
        lat: the latitude
        time_to: a dictionary mapping each state to the time to get to the state
        bat_level: the current battery level of the state
        heading: the heading used to GET to this state
        thrust: the thrust applied to GET to this state
        adjacent: a list of all points (lon, lat), that are adjacent to this point

    """
    thrusts = [0, 0.25, 0.5, 0.75, 1.0]
    # matrices = [], an array of matrices, one for each direction
    u_curr_func, v_curr_func = None, None
    dlon, dlat = None, None

    def __init__(self, lon, lat, time_to, bat_level, heading=None, thrust=None):
        self.lon = lon
        self.lat = lat
        self.time_to = time_to
        self.bat_level = bat_level
        self.heading = heading
        self.thrust = thrust
        self.adjacent = self.adjacent_points()

    def adjacent_points(self):
        # TODO: potentially consider 16 states instead of 8 states
        dlon, dlat = self.dlon, self.dlat
        assert dlon is not None and dlat is not None, "need to set dlon and dlat first"
        offsets = [(0, dlat), (dlon, 0), (dlon, dlat), (0, -dlat), (-dlon, 0),
                   (-dlon, -dlat), (dlon, -dlat), (-dlon, dlat)]
        for lon_offset, lat_offset in offsets:
            yield self.lon + lon_offset, self.lat + lat_offset

    def neighbors(self, thrust_granularity=0.1):
        for lon, lat in self.adjacent:
            # pdb.set_trace()
            thrust, heading, time = self.find_min_thrust(lon, lat)
            if thrust is None:
                continue
            # TODO: potentially add data structure that maps each state to the heading/thrust needed to get there.
            bat_level = self.change_bat_level(thrust, time)
            yield time, State(lon=lon, lat=lat, time_to=self.time_to, bat_level=bat_level, heading=heading,
                              thrust=thrust)
            thrust += thrust_granularity

    def __lt__(self, other):
        return self.time_to[self] < self.time_to[other]

    def __hash__(self):
        return hash((self.lon, self.lat, self.bat_level))

    def __eq__(self, other):
        if abs(self.lon - other.lon) > 0.01 or abs(self.lat - other.lat) > 0.01:
            return False

        if self.bat_level is None or other.bat_level is None:
            return True
        else:
            return other.bat_level == self.bat_level

    @classmethod
    def set_dlon_dlat(cls, lon, lat, target_lon, target_lat):
        """ Sets the dlon and dlat variables of the class appropriately.

        Args:
            lon: longitude, a float
            lat: latitude, a float
            target_lon: the target longitude, a float
            target_lat: the target latitude, a float
        Returns:
            min_thrust, heading, time
        """
        # TODO: address case when points are too close, add keyword arguments
        starting_dlon, starting_dlat = 0.03, 0.03
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

    def find_min_thrust(self, lon, lat, matrix=None, v_min=0.1, distance=None, max_thrust=0.5):
        """ Finds the minimum thrust to go from the current state to the given lon, lat

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

        # TODO: figure out what values for v_min and max_thrust should be used

        def normalize(v):
            return v / math.sqrt(np.sum(v ** 2))

        # TODO: precompute in the future
        # Step 1:  Find the distance between the two points
        dlon = lon - self.lon
        dlat = lat - self.lat
        distance = math.sqrt(dlon * dlon + dlat * dlat)

        # Step 2: Find the current velocity vector (in middle of distance),
        # w.r.t basis with parallel vector (direction of motion) and perpendicular vector
        middle_lon = self.lon + dlon / 2
        middle_lat = self.lat + dlat / 2

        # TODO: currently doesn't take in the time, should address
        u_curr = self.u_curr_func(ca.vertcat(middle_lat, middle_lon))
        v_curr = self.v_curr_func(ca.vertcat(middle_lat, middle_lon))

        parallel_basis_vector = normalize(np.array([dlon, dlat]))
        perp_basis_vector = normalize(np.array([dlat, -dlon]))

        # TODO: precompute in the future
        matrix = np.array([parallel_basis_vector, perp_basis_vector])

        # change basis of vectors w.r.t normalized basis in direction of motion
        change_of_basis = np.linalg.inv(matrix)
        current_vector = np.matmul(np.array([u_curr, v_curr]), change_of_basis)
        curr_parallel, curr_perp = current_vector[0], current_vector[1]

        # Step 4: Set the u_perp to be the negative of the current_perp to counteract it
        u_perp = -curr_perp

        # Step 5: Set magnitude of d u_parallel = max(v_min - c_parallel, 0), heading to be direction between points.
        u_parallel = max(v_min - curr_parallel, 0)

        # Step 6: If u_perp is more than possible thrust, failure
        if abs(u_perp) > max_thrust or abs(u_parallel) > max_thrust:
            return None, None, None

        # Step 7: Find the TIME = distance / (u_parallel + current_parallel)
        time = distance / (u_parallel + curr_parallel)

        # Step 8: Go back to original basis, i.e. in terms of u and v
        u_vector = np.array([u_parallel, u_perp])
        u_dir = np.matmul(u_vector, matrix)

        # Step 9: Find the thrust and heading
        thrust_array = Planner.transform_u_dir_to_u(self=None, u_dir=u_dir)
        thrust, heading = thrust_array[0], thrust_array[1]

        return thrust, heading, time

    def change_bat_level(self, thrust, time):
        # TODO: change the battery level
        return self.bat_level

    @classmethod
    def set_fieldset(cls, problem, type='bspline'):
        cls.u_curr_func, cls.v_curr_func = simulation_utils.get_interpolation_func(
            fieldset=problem.fieldset, type=type, fixed_time_index=problem.fixed_time_index)