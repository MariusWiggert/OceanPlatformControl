import numpy as np
from ocean_navigation_simulator.utils import plotting_utils
import bisect


class Planner:
    """ All Planners should inherit this class

    Attributes:
        problem:
            The Problem to solve containing the vector field, x_0, and x_T

        specific_settings:
            A configuration of the settings specific for that Planner (see respective Planner docstring)

        conv_m_to_deg       constant used to transform from m to deg lat, lon.
    """

    def __init__(self, problem, specific_settings, conv_m_to_deg):

        # extract relevant aspects from the problem
        self.x_0 = np.array(problem.x_0)
        self.x_T = np.array(problem.x_T)
        self.dyn_dict = problem.dyn_dict
        self.problem = problem
        # self.fixed_time = problem.fixed_time

        # Note: managing the forecast fieldsets is done in the simulator
        self.forecast_data_source = None
        self.updated_forecast_source = True

        # initialize vectors for open_loop control
        self.times, self.x_traj, self.contr_seq = None, None, None

        # saving the planned trajectories for inspection purposes
        self.planned_trajs = []

        self.conv_m_to_deg = conv_m_to_deg
        self.specific_settings = specific_settings

    def plan(self, x_t, trajectory=None):
        """ Runs the planner on the problem set at initialization.
        Inputs:
        x_t:
            The current state (lat, lon, bat, time).
        trajectory (Optional):
            The trajectory up to now which can be used by the planner to infer information e.g. by fitting a GP
        """
        raise NotImplementedError()

    def get_u_from_vectors(self, state, ctrl_vec='xy'):
        """ Indexing into the planned open_loop control sequence using the time from state.
        Input Params:
        - state         1D vector with [lat, lon, battery_level, time]
        - ctrl_vec      string of 'xy' if the entries in the control vector are u_x and u_y
                        and 'dir' if the entries are [u, angle]
        """
        # an easy way of finding for each time, which index of control signal to apply
        idx = bisect.bisect_right(self.times, state[3]) - 1
        if idx == len(self.times) - 1:
            idx = idx - 1
            print("WARNING: continuing using last control although not planned as such")

        # extract right element from ctrl vector
        u_out = np.array([[self.contr_seq[0, idx]], [self.contr_seq[1, idx]]])
        if ctrl_vec == 'xy':
            # transform to thrust & angle
            u_out = self.transform_u_dir_to_u(u_dir=u_out)
        elif ctrl_vec != 'dir':
            raise ValueError('ctrl_vec must be either xy or dir')
        return u_out

    def update_forecast_dicts(self, new_forecast_dicts):
        """ Makes sure the forecast fieldset is defined and updates it if a new one is given.
        Input: new_forecast_dicts:
            A list of dicts containing the infos to the newest available forecast/hindcasts on which the planner should operate.
        """
        if new_forecast_dicts is not None:
            print("updating forecast dicts")
            self.forecast_data_source = new_forecast_dicts
            self.updated_forecast_source = True
        else:
            if self.forecast_data_source is None:
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

    def plot_2D_traj(self):
        plotting_utils.plot_2D_traj(self.x_traj)

    def plot_ctrl_seq(self):
        plotting_utils.plot_opt_ctrl(self.times[:-1], self.contr_seq)


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

