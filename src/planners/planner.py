import numpy as np


class Planner:
    """ All Planners should inherit this class

    Attributes:
        problem:
            The Problem to solve containing the vector field, x_0, and x_T

        settings:
            A configuration of the settings of the Planner. Expects the keys below:

            - 'conv_m_to_deg'       # a fixed factor converting m to degree only working far from the pole
            - 'int_pol_type'        # the underlying interpolation type used by the planner to access the current fields

            TODO: add which, if any, other keys could go in the settings dictionary.
    """

    def __init__(self, problem, settings):

        if settings is None:
            settings = {'conv_m_to_deg': 111120., 'int_pol_type': 'bspline'}
        self.problem = problem
        self.settings = settings

    def get_next_action(self, state):
        """ Returns (thrust, header) for the next timestep.

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

    def get_waypoints(self):
        """ Returns waypoints to be used for the waypoint tracking controller.

        Returns:
            An array containing the waypoints for the tracking controller.
        TODO: decide on interface for waypoint tracking!
        """
        raise NotImplementedError()

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

