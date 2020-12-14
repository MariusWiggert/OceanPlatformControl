from src.utils.classes import *
import math


class StraightLinePlanner(Planner):
    """ Straight Line Actuation """

    def __init__(self, problem,
                 settings=None,
                 t_init=806764., n=100, mode='open-loop'):
        # Set all the attributes of the Planner class
        Planner.__init__(self, problem, settings, t_init, n, mode)
        # self.dt = self.T_init / self.N
        self.dt = 116
        self.first_time = True

    def get_next_action(self, state, rel_time):
        """ Currently returns velocity in u and v direction, TODO: return (thrust, header) """

        lon, lat = state[0], state[1]

        # The first time the code is run, the data is [lon, lat]
        # Every later time, the data is [[lon], [lat]]
        # TODO: update temporary fix below
        if not self.first_time:
            lon, lat = lon[0], lat[0]
        self.first_time = False

        lon_target, lat_target = self.problem.x_T[0], self.problem.x_T[1]

        dlon = lon_target - lon
        dlat = lat_target - lat
        mag = math.sqrt(dlon * dlon + dlat * dlat)

        # go there full power
        u_out = np.array([[dlon / mag], [dlat / mag]]) * self.problem.u_max
        return u_out



