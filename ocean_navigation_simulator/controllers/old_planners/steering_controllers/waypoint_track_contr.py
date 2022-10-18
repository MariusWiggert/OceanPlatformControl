import matplotlib.pyplot as plt
import numpy as np


class WaypointTrackingController:
    """Base Class for controllers tracking a set of waypoints.
    A waypoint consists of [lat, lon, time]
    """

    def __init__(self, traj_data=None):
        self.waypoints = None
        self.problem = None
        self.states_traveled = []  # a list of all the states traveled
        self.actuating_towards = (
            []
        )  # a list of each (lon, lat) point we are actuating to at each time step
        self.traj_data = traj_data  # the trajectory data of the

    def set_waypoints(self, waypoints, problem):
        # TODO: needs to be updated so that it is general and also a MPC style tracking controller is possible
        """Changes the waypoints, and resets the state of the Tracking Controller

        Args:
            waypoints: A list of waypoints
            problem: The problem we are solving, not applicable to all controllers

        Returns:
            None
        """
        raise NotImplementedError()

    def replan(self, state):
        """Runs a planning loop in case the tracking controller has one (e.g. an MPC Tracking Controller)
        Does not have to be implemented by the child class if get_next_action contains all the logic required.

        Args:
            state:
                A four element list describing the current state, i.e. [[lon],[lat], [battery_level], [time]]. Note each
                 nested variable is a float.
        Returns:
            None
        """
        return None

    def get_next_action(self, state, trajectory):
        """Returns (thrust, header) for the next timestep

        Args:
            state:
                A four element list describing the current state, i.e. [[lon],[lat], [battery_level], [time]]. Note each
                 nested variable is a float.

        Returns:
            An array containing the thrust and heading, i.e. array([thrust], [heading]).
        """
        raise NotImplementedError()

    def show_traj_waypoints(self, with_currents=True):
        """Shows the PLANNED trajectory"""
        if not self.waypoints:
            print("failure")
            return
        x_labels = list(map(lambda x: x[0], self.waypoints))
        y_labels = list(map(lambda x: x[1], self.waypoints))
        if with_currents and self.problem is not None:
            pset = p.ParticleSet.from_list(
                fieldset=self.problem.fieldset,
                # the fields on which the particles are advected
                pclass=p.ScipyParticle,
                # the type of particles (JITParticle or ScipyParticle)
                lon=x_labels,  # a vector of release longitudes
                lat=y_labels,  # a vector of release latitudes
            )

            pset.show(field="vector", show_time=self.problem.fixed_time_index)
        else:
            plt.scatter(x_labels, y_labels)
            plt.annotate("x_0", (x_labels[0], y_labels[0]))
            plt.annotate("x_T", (x_labels[-1], y_labels[-1]))
            plt.show()

    def show_actual_trajectory(self, with_currents=True):
        """Shows the ACTUAL, i.e. simulated, trajectory in the same plots as the
        planned trajectory. Note the black lines show which waypoint the TTC is actuating to.

        Set with_currents to False to see the actuated path compared to the planned path
        """
        x_labels = list(map(lambda x: x[0], self.states_traveled))
        y_labels = list(map(lambda x: x[1], self.states_traveled))
        # if with_currents:
        #     pset = p.ParticleSet.from_list(fieldset=self.problem.fieldset,
        #                                    # the fields on which the particles are advected
        #                                    pclass=p.ScipyParticle,
        #                                    # the type of particles (JITParticle or ScipyParticle)
        #                                    lon=x_labels,  # a vector of release longitudes
        #                                    lat=y_labels,  # a vector of release latitudes
        #                                    )
        #
        #     pset.show(field='vector', show_time=self.problem.fixed_time_index)
        # else:
        #     fig, ax = plt.subplots()
        #     x_labels = list(map(lambda x: x[0], self.waypoints))
        #     y_labels = list(map(lambda x: x[1], self.waypoints))
        #     ax.scatter(x_labels, y_labels, c='coral', label="Planned Waypoints")
        #     x_labels = list(map(lambda x: x[0], self.states_traveled))
        #     y_labels = list(map(lambda x: x[1], self.states_traveled))
        #     ax.scatter(x_labels, y_labels, c='lightblue', label="Actual Waypoints")
        #     plt.annotate('x_0', (x_labels[0], y_labels[0]))
        #     plt.annotate('x_T', (x_labels[-1], y_labels[-1]))
        #     for x1, x2 in zip(self.states_traveled, self.actuating_towards):
        #         plt.arrow(x=x1[0], y=x1[1], dx=x2[0] - x1[0], dy=x2[1] - x1[1], width=.0006)
        #     ax.legend()
        #     ax.grid(True)
        #     plt.show()

    @staticmethod
    def transform_u_dir_to_u(u_dir):
        """Transforms the given u and v velocities to the corresponding heading and thrust.

        Args:
            u_dir:
                An nested array containing the u and velocities. More rigorously:
                array([[u velocity (longitudinal)], [v velocity (latitudinal)]])

        Returns:
            An array containing the thrust and heading, i.e. array([thrust], [heading]).
        """
        thrust = np.sqrt(
            u_dir[0] ** 2 + u_dir[1] ** 2
        )  # Calculating thrust from distance formula on input u
        heading = np.arctan2(u_dir[1], u_dir[0])  # Finds heading angle from input u
        return np.array([thrust, heading])
