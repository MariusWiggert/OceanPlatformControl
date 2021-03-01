import bisect

from src.planners.planner import Planner
from src.problem_set import ProblemSet
from src.utils import simulation_utils
import heapq
import math
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time as t
import pdb

from src.utils.in_bounds_utils import InBounds


class AStarPlanner(Planner):
    """Discretize the graph and run A Star

    Attributes:
        dt:
            A float giving the time, in seconds, between queries. see Planner class for the rest of the attributes.
        path:
            A list of lon, lat coordinates for the waypoints
        battery_levels:
            The battery level at each waypoint
        control:
            List of thrusts/headings as calculated by find_min_thrust
        failure:
            If we failed.
    """

    def __init__(self, problem,
                 settings=None,
                 t_init=None, n=100, mode='open-loop', dt=10.):
        super().__init__(problem, settings, t_init, n, mode)
        self.dt = dt
        self.path = []
        self.battery_levels = []
        self.times = []
        self.control = []
        self.failure = False
        start_time = t.time()
        self.a_star()
        end_time = t.time()
        print("TIME ELAPSED: ", end_time - start_time)

    def show_trajectory(self):
        if not self.failure:
            x_labels = list(map(lambda x: x[0], self.path))
            y_labels = list(map(lambda x: x[1], self.path))
            plt.scatter(x_labels, y_labels)
            plt.annotate('x_0', (x_labels[0], y_labels[0]))
            plt.annotate('x_T', (x_labels[-1], y_labels[-1]))
            plt.show()
        else:
            print('failure')

    def get_next_action(self, state):
        """Grab the next action. TODO: USE THE TTC """
        assert len(self.times) == len(self.control)
        assert sorted(self.times) == self.times, "The times should be in sorted order: " + str(self.times)
        idx = bisect.bisect_left(self.times, state[3])
        print("Index: {}".format(idx))
        if idx == len(self.times):
            idx -= 1
        return self.control[idx]

    def heuristic(self, state, target_lon, target_lat, max_velocity=0.6):
        """ Return the minimum time between the given state and the end target """
        distance = math.sqrt((state.lon - target_lon)**2 + (state.lat - target_lat)**2)
        return distance * 111120. / max_velocity

    def a_star(self):
        """ Discretize the graph and find the time optimal path using A Star
        """
        # data structures
        time_to = {}
        edge_to = {}
        seen = {}   # we will map each (lon, lat) to the (time, bat_level) used to get there.
        pq = []

        # set up problem
        lon, lat = self.problem.x_0[0], self.problem.x_0[1]
        target_lon, target_lat = self.problem.x_T[0], self.problem.x_T[1]

        # set the state class variables
        State.set_dlon_dlat(lon, lat, target_lon, target_lat)
        State.set_fieldset(self.problem)
        State.time_to = time_to
        State.problem = self.problem
        State.heuristic = lambda s: self.heuristic(s, target_lon, target_lat)
        State.in_bounds = InBounds(self.problem.fieldset)
        # State.heuristic = lambda s: 0

        starting_state = State(lon=lon, lat=lat, bat_level=1.0)
        pq.append(starting_state)
        time_to[starting_state] = 0
        edge_to[starting_state] = starting_state
        state = starting_state
        end_state = State(lon=target_lon, lat=target_lat, bat_level=None)

        # run Dijkstra's
        max_time = 120
        starting_time = t.time()
        while not state == end_state:
            if not pq or starting_time + max_time < t.time():
                self.failure = True
                return
            state = heapq.heappop(pq)

            if (state.lon, state.lat) not in seen:
                seen[(state.lon, state.lat)] = (time_to[state], state.bat_level)
            else:
                best_time, best_bat_level = seen[(state.lon, state.lat)]
                seen[(state.lon, state.lat)] = min(best_time, time_to[state]), max(best_bat_level, state.bat_level)
            for time, neighbor_state in state.neighbors():
                if (neighbor_state.lon, neighbor_state.lat) in seen:
                    recorded_time, recorded_bat_level = seen[(neighbor_state.lon, neighbor_state.lat)]

                    # if we are considering a neighbor that is worse than seen before
                    if neighbor_state in time_to and time_to[neighbor_state] >= recorded_time \
                            and neighbor_state.bat_level <= recorded_bat_level:
                        continue
                if neighbor_state not in pq:
                    pq.append(neighbor_state)

                new_time = time + time_to[state]
                if neighbor_state not in time_to or new_time < time_to[neighbor_state]:
                    time_to[neighbor_state] = new_time
                    edge_to[neighbor_state] = state
            heapq.heapify(pq)

        # extract the waypoints
        self.times, self.path, self.control, self.battery_levels = [], [], [], []
        while state != starting_state:
            self.battery_levels.insert(0, state.bat_level)
            self.path.insert(0, (state.lon, state.lat))
            self.times.insert(0, time_to[state])
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
    u_curr_func, v_curr_func = None, None
    dlon, dlat = None, None
    time_to = None
    problem = None
    heuristic = None
    in_bounds = None

    def __init__(self, lon, lat, bat_level, heading=None, thrust=None):
        self.lon = lon
        self.lat = lat
        self.bat_level = bat_level
        self.heading = heading
        self.thrust = thrust
        self.adjacent = self.adjacent_points()

    def adjacent_points(self):
        dlon, dlat = self.dlon, self.dlat
        assert dlon is not None and dlat is not None, "need to set dlon and dlat first"
        offsets = [(0, dlat), (dlon, 0), (dlon, dlat), (0, -dlat), (-dlon, 0),
                   (-dlon, -dlat), (dlon, -dlat), (-dlon, dlat)]
        for lon_offset, lat_offset in offsets:
            point = (self.lon + lon_offset, self.lat + lat_offset)
            if State.in_bounds.in_ocean(point):
                yield point

    def neighbors(self):
        for lon, lat in self.adjacent:
            for thrust, heading, time in self.actuate_towards(lon, lat):
                bat_level = self.change_bat_level(thrust, time)
                if bat_level >= 0.1:
                    yield time, State(lon=lon, lat=lat, bat_level=bat_level, heading=heading, thrust=thrust)

    def __lt__(self, other):
        return self.time_to[self] + State.heuristic(self) < self.time_to[other] + State.heuristic(other)

    def __hash__(self):
        return hash((self.lon, self.lat, self.bat_level))

    def __eq__(self, other):
        if abs(self.lon - other.lon) > 0.01 or abs(self.lat - other.lat) > 0.01:
            return False

        if self.bat_level is None or other.bat_level is None:
            return True
        else:
            return self.bat_level_round(other.bat_level) == self.bat_level_round(self.bat_level)

    def bat_level_round(self, bat_level):
        return math.floor(bat_level * 20) / 20

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
        starting_dlon, starting_dlat = 0.05, 0.05
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

    def actuate_towards(self, lon, lat, matrix=None, v_min=0.1, distance=None):
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

        # TODO: figure out what values for v_min and max_thrust should be used

        def normalize(v):
            return v / math.sqrt(np.sum(v ** 2))

        max_thrust = self.problem.dyn_dict['u_max']

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
        u_parallel_min = max(v_min - curr_parallel, 0)

        # Step 6: If u_perp is more than possible thrust, failure
        if math.sqrt(u_perp ** 2 + u_parallel_min ** 2) > max_thrust:
            return

        u_parallel_max = math.sqrt(max_thrust**2 - u_perp**2)
        # u_parallels = list(np.arange(u_parallel_min, u_parallel_max - 0.1, 0.1)) + [u_parallel_max]

        # Can potentially consider more than one thrust
        for u_parallel in [u_parallel_min]:

            # Step 7: Find the TIME = distance / (u_parallel + current_parallel)
            # TODO: replace hard coded constant with settings data reference
            time = distance * 111120. / (u_parallel + curr_parallel)

            # Step 8: Go back to original basis, i.e. in terms of u and v
            u_vector = np.array([u_parallel, u_perp])
            u_dir = np.matmul(u_vector, matrix)

            # Step 9: Find the thrust and heading
            thrust_array = Planner.transform_u_dir_to_u(self=None, u_dir=u_dir)
            thrust, heading = thrust_array[0], thrust_array[1]

            yield thrust, heading, time

    def change_bat_level(self, thrust, time):
        # return self.bat_level
        bat_level_delta = self.problem.dyn_dict['charge'] - self.problem.dyn_dict['energy'] * (thrust ** 3)
        return min(self.bat_level + time * bat_level_delta, 1)

    @classmethod
    def set_fieldset(cls, problem, type='bspline'):
        cls.u_curr_func, cls.v_curr_func = simulation_utils.get_interpolation_func(
            fieldset=problem.fieldset, type=type, fixed_time_index=problem.fixed_time_index)
