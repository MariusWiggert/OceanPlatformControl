import numpy as np


class Planner:
    """ All Planners should inherit this class

    Attributes:
        problem:
            The Problem to solve containing the vector field, x_0, and x_T
        T_init:
            A fixed amount of time for the trip, not applicable to all planners.
        N:
            The number of decision variables in the trajectory
        settings:
            A configuration of the settings of the Planner. Expects the keys below:

            - 'conv_m_to_deg'
            - 'int_pol_type'
            - 'dyn_constraints'

            TODO: add which, if any, other keys could go in the settings dictionary.
    """

    def __init__(self, problem, settings, t_init, n, mode='open-loop', fixed_time_index=None):

        if settings is None:
            settings = {'conv_m_to_deg': 111120., 'int_pol_type': 'bspline', 'dyn_constraints': 'ef'}
        self.problem = problem
        self.T_init = t_init
        self.N = n
        self.settings = settings
        self.mode = mode

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
        return "Planner(mode: {0}, problem: {1})".format(self.mode, self.problem)

    def transform_u_dir_to_u(self, u_dir):
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

