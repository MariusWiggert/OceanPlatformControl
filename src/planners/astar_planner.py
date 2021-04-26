import bisect

from src.State import State
from src.planners.planner import Planner
from src.problem_set import ProblemSet
from src.tracking_controllers.minimumThrustController import MinimumThrustController
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

    def __init__(self, problem, settings=None):
        super().__init__(problem, settings)

        self.path = []
        self.battery_levels = []
        self.times = []
        self.control = []
        self.failure = False

        # run the A-Star algorithm for path planning
        start_time = t.time()
        self.a_star()
        end_time = t.time()
        self.TTC = MinimumThrustController(self.path, self.times, self.problem)
        print("TIME ELAPSED: ", end_time - start_time)

    def get_next_action(self, state):
        """ Grab the next action by actuating to the nearest/best waypoint with the minimum thrust logic
        Args: state
        Returns: (thrust, header)
        """
        return self.TTC.get_next_action(state)

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
        seen = {}  # we will map each (lon, lat) to the (time, bat_level) used to get there
        # priority queue
        pq = []

        # set up problem
        lon, lat = self.problem.x_0[0], self.problem.x_0[1]
        target_lon, target_lat = self.problem.x_T[0], self.problem.x_T[1]

        # set the state class variables
        State.set_dlon_dlat(0.03, lon, lat, target_lon, target_lat)
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