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
