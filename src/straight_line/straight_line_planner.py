from src.utils.classes import *
import math


class StraightLinePlanner(Planner):
    """ Straight Line Actuation """

    def __init__(self, problem,
                 settings=None,
                 t_init=806764., n=100, mode='open-loop'):
        # Set all the attributes of the Planner class
        Planner.__init__(self, problem, settings, t_init, n, mode)
        self.dt = self.T_init / self.N

    def get_next_action(self, state):
        """ Go in the direction of the header """

        lon, lat = state[0][0], state[1][0]

        lon_target, lat_target = self.problem.x_T[0], self.problem.x_T[1]

        dlon = lon_target - lon
        dlat = lat_target - lat
        mag = math.sqrt(dlon * dlon + dlat * dlat)

        # go there full power
        u_dir = np.array([[dlon / mag], [dlat / mag]]) * self.problem.u_max
        u_out = super().transform_u_dir_to_u(u_dir=u_dir)
        return u_out



