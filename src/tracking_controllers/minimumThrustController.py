from src.State import State
from src.tracking_controllers.waypoint_track_contr import WaypointTrackingController
import numpy as np
import math, bisect
import matplotlib.pyplot as plt
import parcels as p
from parcels.plotting import plotparticles


class MinimumThrustController(WaypointTrackingController):
    """ Simple proportional controller:
        Outside a radius r from the waypoint: actuate full power
        Inside the radius actuate: linearly decreasing actuation
    """

    def __init__(self, path, times, problem):
        super().__init__()
        self.path = path
        self.times = times
        if not self.path:
            self.failure = True
        else:
            self.failure = False
        self.problem = problem
        self.states_traveled = []
        self.actuating_towards = []

    def show_planned_trajectory(self, with_currents=True):
        """ Shows the PLANNED trajectory """
        if self.failure:
            print('failure')
            return
        x_labels = list(map(lambda x: x[0], self.path))
        y_labels = list(map(lambda x: x[1], self.path))
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
