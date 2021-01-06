from src.utils.classes import *
import math


class StraightLinePlanner(Planner):
    """Straight Line Actuation

    Attributes:
        dt:
            A float giving the time, in seconds, between queries.
        see Planner class for the rest of the attributes.
    """

    def __init__(self, problem,
                 settings=None,
                 t_init=None, n=100, mode='open-loop'):
        Planner.__init__(self, problem, settings, t_init, n, mode)
        # self.dt = self.T_init / self.N
        self.dt = 10.

    def get_next_action(self, state):
        """Go in the direction of the target with full power.

        Args:
            state:
                A four element list describing the current state, i.e. [[lon],[lat], [battery_level], [time]]. Note each
                 nested variable is a float.

        Returns:
            An array containing the thrust and heading, i.e. array([thrust], [heading]).
        """

        lon, lat = state[0][0], state[1][0]
        lon_target, lat_target = self.problem.x_T[0], self.problem.x_T[1]

        dlon = lon_target - lon
        dlat = lat_target - lat
        mag = math.sqrt(dlon * dlon + dlat * dlat)

        # go there full power
        u_dir = np.array([[dlon / mag], [dlat / mag]])
        u_out = super().transform_u_dir_to_u(u_dir=u_dir)
        return u_out
