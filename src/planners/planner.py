import numpy as np

from src.utils import hycom_utils


class Planner:
    """ All Planners should inherit this class

    Attributes:
        problem:
            The Problem to solve containing the vector field, x_0, and x_T

        gen_settings:
            A configuration of the general settings of the Planner. Expects the keys below:

            - 'conv_m_to_deg'       # a fixed factor converting m to degree only working far from the pole
            - 'int_pol_type'        # the underlying interpolation type used by the planner to access the current fields

        specific_settings:
            A configuration of the settings specific for that Planner (see respective Planner docstring)
    """

    def __init__(self, problem, gen_settings, specific_settings):

        # extract relevant aspects from the problem
        self.x_0 = problem.x_0
        self.x_T = problem.x_T
        self.dyn_dict = problem.dyn_dict
        self.fixed_time = problem.fixed_time

        # Note: managing the forecast fieldsets is done in the simulator
        self.cur_forecast_file = None

        self.gen_settings = gen_settings
        self.specific_settings = specific_settings

    def plan(self, x_t, new_forecast_file=None, trajectory=None):
        """ Runs the planner on the problem set at initialization.
        Inputs:
        x_t:
            The current state (lat, lon, bat, time).
        new_forecast_file (Optional):
            A string containing the path to the newest available forecast on which the planner should operate.
            If none is supplied then the last one is used.
        trajectory (Optional):
            The trajectory up to now which can be used by the planner to infer information e.g. by fitting a GP
        """
        raise NotImplementedError()

    def update_forecast_file(self, new_forecast_file):
        """ Makes sure the forecast fieldset is defined and updates it if a new one is given."""
        if new_forecast_file is not None:
            print("updating forecast file")
            self.cur_forecast_file = new_forecast_file
        else:
            if self.cur_forecast_file is None:
                raise ValueError('No forecast file is available.')

    def get_next_action(self, state):
        """ Using this function is equivalent to open-loop control without a waypoint tracker.
        It returns (thrust, header) for the next timestep.

        Args:
            state:
                A four element list describing the current state, i.e. [[lon],[lat], [battery_level], [time]]. Note each
                 nested variable is a float.

        Returns:
            An array containing the thrust and heading, i.e. array([thrust], [heading]).
        """
        raise NotImplementedError()

    def __repr__(self):
        return "Planner(x_0: {0}, x_T: {1})".format(self.x_0, self.x_T)

    def get_waypoints(self):
        """ Returns waypoints to be used for the waypoint tracking controller.

        Returns:
            An array containing the waypoints for the tracking controller.
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
            Heading is in radians.
        """
        thrust = np.sqrt(u_dir[0]**2 + u_dir[1]**2)  # Calculating thrust from distance formula on input u
        heading = np.arctan2(u_dir[1], u_dir[0])  # Finds heading angle from input u
        return np.array([thrust, heading])

    @staticmethod
    def replan(self, state=None):
        """ Dummy function so the planner class can be used for closed-loop control testing."""
        return None

    @staticmethod
    def set_waypoints(self, waypoints=None, problem=None):
        """ Dummy function so the planner class can be used for closed-loop control testing."""
        return None

