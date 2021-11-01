from ocean_navigation_simulator.planners.planner import Planner
import math
import numpy as np


class PassiveFloating(Planner):
    """Passively floating of the platform. For comparison reasons.
    """

    def __init__(self, problem, gen_settings, specific_settings):
        print("Instantiating passively floating (do nothing) controller.")
        super().__init__(problem, gen_settings, specific_settings)

    def plan(self, x_t, new_forecast_file=None, trajectory=None):
        """This planner doesn't need any re-planning."""
        return

    def get_next_action(self, x_t):
        """Just return 0. """

        # go there full power
        u_dir = np.array([[0., 0.]]).T
        return u_dir

    def get_waypoints(self):
        return [[0, 0, 0]]
