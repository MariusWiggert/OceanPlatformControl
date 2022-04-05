from ocean_navigation_simulator import Problem
from ocean_navigation_simulator.controllers import simulator_data
from ocean_navigation_simulator.controllers.controller import Controller
from ocean_navigation_simulator.controllers.utils import transform_u_dir_to_u
import math
import numpy as np


class StraightLineController(Controller):
    """
    Straight Line, Full-power Actuation towards the goal (meant as a baseline)
    """

    def __init__(self, problem: Problem, specific_settings, conv_m_to_deg):
        """
        StraightLineController constructor
        Args:
            problem:
            specific_settings:
            conv_m_to_deg:
        TODO: change how it intakes the Problem
        """
        self.x_0 = np.array(problem.x_0)
        self.x_T = np.array(problem.x_T)

        self.problem = problem

        self.conv_m_to_deg = conv_m_to_deg
        self.specific_settings = specific_settings

    def get_action(self, observation: simulator_data.SimulatorObservation) -> np.ndarray:
        """
        Go in the direction of the target with full power.
        """
        x_t = observation.platform_observation.x_t

        lon, lat = x_t[0][0], x_t[1][0]
        lon_target, lat_target = self.x_T[0], self.x_T[1]

        dlon = lon_target - lon
        dlat = lat_target - lat
        mag = math.sqrt(dlon * dlon + dlat * dlat)

        # go there full power
        u_dir = np.array([[dlon / mag], [dlat / mag]])
        u_out = transform_u_dir_to_u(u_dir=u_dir)
        return u_out

    def get_waypoints(self) -> list:
        start = self.problem.x_0[:2] + [self.problem.x_0[-1]]
        end = self.problem.x_T + [self.problem.x_0[-1] + 3600 * self.specific_settings['x_T_time_ahead_in_h']]
        return [start, end]
