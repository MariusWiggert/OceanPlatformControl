from ocean_navigation_simulator.planners.planner import Planner
import math
import numpy as np


class StraightLinePlanner(Planner):
    """Straight Line, Full-power Actuation towards the goal (meant as a baseline)
    See Planner class for attributes.
    """

    def __init__(self, problem, specific_settings, conv_m_to_deg):
        super().__init__(problem, specific_settings, conv_m_to_deg)

    def plan(self, x_t, trajectory=None):
        """This planner doesn't need any re-planning."""
        return None

    def get_next_action(self, x_t):
        """Go in the direction of the target with full power. See superclass for args and return value. """

        lon, lat = x_t[0][0], x_t[1][0]
        lon_target, lat_target = self.x_T[0], self.x_T[1]

        dlon = lon_target - lon
        dlat = lat_target - lat
        mag = math.sqrt(dlon * dlon + dlat * dlat)

        # go there full power
        u_dir = np.array([[dlon / mag], [dlat / mag]])
        u_out = super().transform_u_dir_to_u(u_dir=u_dir)
        return u_out

    def get_waypoints(self):
        start = self.problem.x_0[:2] + [self.problem.x_0[-1]]
        end = self.problem.x_T + [self.problem.x_0[-1] + 3600 * self.specific_settings['x_T_time_ahead_in_h']]
        return [start, end]
