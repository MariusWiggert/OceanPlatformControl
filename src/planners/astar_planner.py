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
import parcels as p
from parcels.plotting import plotparticles

import pdb

from src.utils.in_bounds_utils import InBounds


class AStarPlanner(Planner):
    """ Discretize the graph and run A Star

    Attributes:
        dt:
            A float giving the time, in seconds, between queries. see Planner class for the rest of the attributes.
        last_waypoint_index:
            The index of the current waypoint we are trying to actuate towards
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
        self.actuating_towards = []
        self.battery_levels = []
        self.times = []
        self.control = []
        self.states_traveled = []
        self.failure = False
        start_time = t.time()
        self.a_star()
        end_time = t.time()
        print("TIME ELAPSED: ", end_time - start_time)

    def show_planned_trajectory(self, with_currents=True):
        """ Shows the PLANNED trajectory """
        if self.failure:
            print('failure')
            return
        x_labels = list(map(lambda x: x[0], self.path))
        y_labels = list(map(lambda x: x[1], self.path))
        if with_currents:
            pset = p.ParticleSet.from_list(fieldset=self.problem.fieldset,  # the fields on which the particles are advected
                                           pclass=p.ScipyParticle,
                                           # the type of particles (JITParticle or ScipyParticle)
                                           lon=x_labels,  # a vector of release longitudes
                                           lat=y_labels,  # a vector of release latitudes
                                           )

            pset.show(field='vector', show_time=self.problem.fixed_time_index)
        else:
            plt.scatter(x_labels, y_labels)
            plt.annotate('x_0', (x_labels[0], y_labels[0]))
            plt.annotate('x_T', (x_labels[-1], y_labels[-1]))
            plt.show()

    def show_actual_trajectory(self, with_currents=True):
        """ Shows the ACTUAL, i.e. simulated, trajectory in the same plots as the
        planned trajectory. Note the black lines show which waypoint the TTC is actuating to.

        Set with_currents to False to see the actuated path compared to the planned path
        """
        x_labels = list(map(lambda x: x[0], self.states_traveled))
        y_labels = list(map(lambda x: x[1], self.states_traveled))
        if with_currents:
            pset = p.ParticleSet.from_list(fieldset=self.problem.fieldset,
                                           # the fields on which the particles are advected
                                           pclass=p.ScipyParticle,
                                           # the type of particles (JITParticle or ScipyParticle)
                                           lon=x_labels,  # a vector of release longitudes
                                           lat=y_labels,  # a vector of release latitudes
                                           )

            pset.show(field='vector', show_time=self.problem.fixed_time_index)
        else:
            fig, ax = plt.subplots()
            x_labels = list(map(lambda x: x[0], self.path))
            y_labels = list(map(lambda x: x[1], self.path))
            ax.scatter(x_labels, y_labels, c='coral', label="Planned Waypoints")
            x_labels = list(map(lambda x: x[0], self.states_traveled))
            y_labels = list(map(lambda x: x[1], self.states_traveled))
            ax.scatter(x_labels, y_labels, c='lightblue', label="Actual Waypoints")
            plt.annotate('x_0', (x_labels[0], y_labels[0]))
            plt.annotate('x_T', (x_labels[-1], y_labels[-1]))
            for x1, x2 in zip(self.states_traveled, self.actuating_towards):
                plt.arrow(x=x1[0], y=x1[1], dx=x2[0] - x1[0], dy=x2[1] - x1[1], width=.0006)
            ax.legend()
            ax.grid(True)
            plt.show()

    def get_next_action(self, state):
        """ Grab the next action by actuating to the nearest/best waypoint with the minimum thrust logic
        Args: state
        Returns: (thrust, header)
        """

        # HELPER FUNCTIONS/CLASSES
        class PotentialWaypoint:
            """ Collection of information on the next waypoint to potentially actuate to.

            Attributes:
                waypoint: the lat, lon of the waypoint
                thrust: the needed thrust to actuate here
                heading: the calculated heading to actuate here
                proposed_vector: the vector between this waypoint and the current state
                planned_vector: the vector between this waypoint and the PLANNED waypoint before it
            """

            def __init__(self, waypoint, thrust, heading, time, proposed_vector, planned_vector):
                self.waypoint = waypoint
                self.thrust = thrust
                self.heading = heading
                self.time = time
                self.proposed_vector = proposed_vector
                self.planned_vector = planned_vector

            def cost(self):
                """ This method will be used as a helper routine to figure out which waypoint to go to next. More
                concretely, this function will quantify how good a waypoint is, where a smaller number is a better
                waypoint. """

                # A waypoint is "good" if the directions of the proposed vector and the planned vector are similar.

                vector_1 = self.planned_vector
                vector_2 = self.proposed_vector
                unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
                unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
                dot_product = np.dot(unit_vector_1, unit_vector_2)
                angle = np.arccos(dot_product)

                return abs(angle)

        def vec(cur_state, next_state):
            """ Creates a vector from the current state to the next state

            Args:
                cur_state: (lon, lat)
                next_state: (lon, lat)
            """
            return [next_state[0] - cur_state[0], next_state[1] - cur_state[1]]

        def mag(u_dir):
            """ Returns the magnitude of the given vector"""

            return math.sqrt(u_dir[1] * u_dir[1] + u_dir[0] * u_dir[0])

        def nearest_waypoint(all_waypoints, curr_wp):
            """ Finds the index of the nearest planned waypoint to where we currently are in the simulation """

            distances_away = np.array([mag(vec(wp, curr_wp)) for wp in all_waypoints])
            return np.argmin(distances_away)

        # extract data from passed in state
        lon, lat, bat_level, time_to_state = state[0][0], state[1][0], state[2], state[3]

        # set the state class variables
        State.set_fieldset(self.problem)
        State.problem = self.problem

        # ACTUATE IN THAT DIRECTION USING MIN THRUST FUNCTION
        cur_state = State(lon=lon, lat=lat, bat_level=bat_level)

        prospective_waypoints = []

        # only consider points after the given waypoint, w.r.t to time
        prune_start = bisect.bisect_left(self.times, time_to_state)
        # prune_start = 0
        index = prune_start + nearest_waypoint(self.path[prune_start:], (lon, lat))
        self.states_traveled.append((lon, lat))
        self.actuating_towards.append(self.path[index])
        # print('NEAREST WAYPOINT IS APPARENTLY ', self.path[index])

        start_index = max(1, index - 3)
        # print("prune start: ", prune_start, " and the nearest index is ", index)
        end_index = min(len(self.path), index + 3)
        for i in range(start_index, end_index):
            waypoint, prev = self.path[i], self.path[i - 1]
            try:
                thrust, heading, time = next(cur_state.actuate_towards(waypoint[0], waypoint[1],
                                                                       print_output=True, use_middle=False,
                                                                       full_send=True, cushion=0.1))

                # consider adding in the time here, and LQR


                # nan if we are actuating to ourself.
                if np.isnan(thrust):
                    continue
                vector = vec((lon, lat), waypoint)
                planned_vector = vec(prev, waypoint)

                prospective_waypoints.append(PotentialWaypoint(waypoint, thrust, heading, time, vector, planned_vector))
            except StopIteration:
                continue

        if not prospective_waypoints:
            self.show_actual_trajectory()

        print("WAYPOINTS FOUND", prospective_waypoints)
        best_waypoint_state = min(prospective_waypoints, key=lambda wp: wp.cost())
        chosen_actuation = (best_waypoint_state.thrust, best_waypoint_state.heading)

        print("AT: {0} \nGOING TO: {1}".format(str(cur_state), str(best_waypoint_state.waypoint)))
        print(chosen_actuation)
        return chosen_actuation[0], chosen_actuation[1]

    def heuristic(self, state, target_lon, target_lat, max_velocity=0.6):
        """ Return the minimum time between the given state and the end target """
        distance = math.sqrt((state.lon - target_lon) ** 2 + (state.lat - target_lat) ** 2)
        return distance * 111120. / max_velocity

    def a_star(self):
        """ Discretize the graph and find the time optimal path using A Star
        """
        # data structures
        time_to = {}
        edge_to = {}
        seen = {}  # we will map each (lon, lat) to the (time, bat_level) used to get there.
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
        done = False
        while not done:
            self.battery_levels.insert(0, state.bat_level)
            self.path.insert(0, (state.lon, state.lat))
            self.times.insert(0, time_to[state])
            self.control.insert(0, (state.thrust, state.heading))
            done = edge_to[state] == state
            state = edge_to[state]

        print("PATH and THRUST/HEADINGS  ", list(zip(self.path, self.control)))


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
        """ Returns a list of all valid that are adjacent in the 8 directions """
        dlon, dlat = self.dlon, self.dlat
        assert dlon is not None and dlat is not None, "need to set dlon and dlat first"
        offsets = [(0, dlat), (dlon, 0), (dlon, dlat), (0, -dlat), (-dlon, 0),
                   (-dlon, -dlat), (dlon, -dlat), (-dlon, dlat)]
        for lon_offset, lat_offset in offsets:
            point = (self.lon + lon_offset, self.lat + lat_offset)
            if State.in_bounds.in_ocean(point):
                yield point

    def neighbors(self):
        """ Returns all the neighboring states in A, along with the time to get there

        Returns:
            time, next state
        """
        for lon, lat in self.adjacent:
            for thrust, heading, time in self.actuate_towards(lon, lat):
                bat_level = self.change_bat_level(thrust, time)
                if bat_level >= 0.1:
                    yield time, State(lon=lon, lat=lat, bat_level=bat_level, heading=heading, thrust=thrust)

    def __lt__(self, other):
        return self.time_to[self] + State.heuristic(self) < self.time_to[other] + State.heuristic(other)

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
        def normalize(v):
            return v / math.sqrt(np.sum(v ** 2))

        max_thrust = self.problem.dyn_dict['u_max'] + cushion

        # TODO: precompute in the future
        # Step 1:  Find the distance between the two points
        dlon = lon - self.lon
        dlat = lat - self.lat
        distance = math.sqrt(dlon * dlon + dlat * dlat)

        if use_middle:
            # Step 2: Find the current velocity vector (in middle of distance),
            # w.r.t basis with parallel vector (direction of motion) and perpendicular vector
            middle_lon = self.lon + dlon / 2
            middle_lat = self.lat + dlat / 2

            # TODO: currently doesn't take in the time, should address
            u_curr = self.u_curr_func(ca.vertcat(middle_lat, middle_lon))
            v_curr = self.v_curr_func(ca.vertcat(middle_lat, middle_lon))
        else:
            u_curr = self.u_curr_func(ca.vertcat(self.lat, self.lon))
            v_curr = self.v_curr_func(ca.vertcat(self.lat, self.lon))

        parallel_basis_vector = normalize(np.array([dlon, dlat]))
        perp_basis_vector = normalize(np.array([dlat, -dlon]))
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
            thrust_array = Planner.transform_u_dir_to_u(u_dir=u_dir)
            thrust, heading = thrust_array[0], thrust_array[1]
            yield thrust, heading, time

    def change_bat_level(self, thrust, time):
        bat_level_delta = self.problem.dyn_dict['charge'] - self.problem.dyn_dict['energy'] * (thrust ** 3)
        return min(self.bat_level + time * bat_level_delta, 1)

    @classmethod
    def set_fieldset(cls, problem, type='bspline'):
        cls.u_curr_func, cls.v_curr_func = simulation_utils.get_interpolation_func(
            fieldset=problem.fieldset, type=type, fixed_time_index=problem.fixed_time_index)
