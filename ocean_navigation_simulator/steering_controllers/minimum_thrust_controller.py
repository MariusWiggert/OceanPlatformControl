from ocean_navigation_simulator.utils.state import State
from ocean_navigation_simulator.steering_controllers import WaypointTrackingController
import numpy as np
import math, bisect
import matplotlib.pyplot as plt
import parcels as p
from parcels.plotting import plotparticles


class MinimumThrustController(WaypointTrackingController):
    """ Continually actuate with the minimum thrust in the direction of the currents,
    """

    def __init__(self, waypoints, problem=None):
        super().__init__()
        if waypoints is not None:
            self.path = list(map(lambda wp: (wp[0], wp[1]), waypoints))
            self.times = list(map(lambda wp: wp[2], waypoints))
        else:
            self.path = []
            self.times = []
        self.problem = problem
        self.states_traveled = []
        self.actuating_towards = []

    def set_waypoints(self, waypoints, problem):
        self.__init__(waypoints=waypoints, problem=problem)

    def get_next_action(self, state):
        """ Returns (thrust, header) for the next timestep
        Args:
            state:
                A four element list describing the current state, i.e. [[lon],[lat], [battery_level], [time]]. Note each
                 nested variable is a float.
        Returns:
            An array containing the thrust and heading, i.e. array([thrust], [heading]).
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
                                                                       print_output=False, use_middle=False,
                                                                       full_send=True, cushion=0.1))

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

        # print("WAYPOINTS FOUND", prospective_waypoints)
        best_waypoint_state = min(prospective_waypoints, key=lambda wp: wp.cost())
        chosen_actuation = (best_waypoint_state.thrust, best_waypoint_state.heading)

        # print("AT: {0} \nGOING TO: {1}".format(str(cur_state), str(best_waypoint_state.waypoint)))
        print(chosen_actuation)
        return chosen_actuation[0], chosen_actuation[1]

    def __str__(self):
        return "Minimum Thrust Controller"
