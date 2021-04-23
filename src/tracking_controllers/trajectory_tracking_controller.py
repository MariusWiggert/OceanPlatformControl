import numpy as np


class WaypointTrackingController:
    """ Base Class for controllers tracking a set of waypoints.
    A waypoint consists of [lat, lon, time]
    """

    def __init__(self):
        # Lis
        self.waypoints = None
        pass


    def get_next_action(self, state):
        """ Returns (thrust, header) for the next timestep

        Args:
            state:
                A four element list describing the current state, i.e. [[lon],[lat], [battery_level], [time]]. Note each
                 nested variable is a float.

        Returns:
            An array containing the thrust and heading, i.e. array([thrust], [heading]).
        """
        raise NotImplementedError()

    def __repr__(self):
        """ """
        return "Planner(problem: {1})".format(self.problem)

    @staticmethod
    def transform_u_dir_to_u(u_dir):
        """ Transforms the given u and v velocities to the corresponding heading and thrust.

        Args:
            u_dir:
                An nested array containing the u and velocities. More rigorously:
                array([[u velocity (longitudinal)], [v velocity (latitudinal)]])

        Returns:
            An array containing the thrust and heading, i.e. array([thrust], [heading]).
        """
        thrust = np.sqrt(u_dir[0]**2 + u_dir[1]**2)  # Calculating thrust from distance formula on input u
        heading = np.arctan2(u_dir[1], u_dir[0])  # Finds heading angle from input u
        return np.array([thrust, heading])
