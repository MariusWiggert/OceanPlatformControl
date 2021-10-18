from ocean_platform_package.src.planners.planner import Planner
import math
import numpy as np


class StraightLinePlanner(Planner):
    """Straight Line, Full-power Actuation towards the goal (meant as a baseline)
    See Planner class for attributes.
    """

    def __init__(self, problem, gen_settings, specific_settings):
        super().__init__(problem, gen_settings, specific_settings)

    def run(self, x_t, new_forecast_file=None, trajectory=None):
        """This planner doesn't need any re-planning."""
        return

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
        raise NotImplementedError
