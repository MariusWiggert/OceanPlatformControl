from abc import ABC

from src.planners.planner import Planner
import math
import numpy as np


class StraightLinePlanner(Planner):
    """Straight Line Actuation

    Attributes:
        dt:
            A float giving the time, in seconds, between queries.
        see Planner class for the rest of the attributes.
    """

    def __init__(self, problem, settings=None, dt=10.):
        super().__init__(problem, settings)
        self.dt = dt

    def get_next_action(self, state):
        """Go in the direction of the target with full power. See superclass for args and return value. """

        lon, lat = state[0][0], state[1][0]
        lon_target, lat_target = self.problem.x_T[0], self.problem.x_T[1]

        dlon = lon_target - lon
        dlat = lat_target - lat
        mag = math.sqrt(dlon * dlon + dlat * dlat)

        # go there full power
        u_dir = np.array([[dlon / mag], [dlat / mag]])
        u_out = super().transform_u_dir_to_u(u_dir=u_dir)
        return u_out

    def get_waypoints(self):
        raise NotImplementedError
