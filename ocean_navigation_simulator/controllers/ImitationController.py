from os.path import dirname, abspath

import numpy as np
import torch

from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.Platform import PlatformAction
from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import HJReach2DPlanner
from ocean_navigation_simulator.ocean_observer.Observer import Observer

TRUE_CURRENT_LENGTH = 5
TTR_MAP_IN_WIDTH = 15
TTR_MAP_IN_WIDTH_DEG = 0.25
TTR_MAP_OUT_WIDTH = 3
TTR_MAP_OUT_WIDTH_DEG = 0.25 / 5


class ImitationController(HJReach2DPlanner):
    """
    Controller using Imitation Learning on te Value Function
    """

    def __init__(self, problem: NavigationProblem, platform_dict):
        """
        StraightLineController constructor
        Args:
            problem: the Problem the controller will run on
        """
        specific_settings = {
            "replan_on_new_fmrc": True,
            "replan_every_X_seconds": None,
            "direction": "multi-time-reach-back",
            "n_time_vector": 100,  # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
            "deg_around_xt_xT_box": 1.0,  # area over which to run HJ_reachability
            "accuracy": "high",
            "artificial_dissipation_scheme": "local_local",
            "T_goal_in_seconds": problem.timeout,
            "use_geographic_coordinate_system": True,
            "progress_bar": True,
            "initial_set_radii": [
                0.1,
                0.1,
            ],  # this is in deg lat, lon. Note: for Backwards-Reachability this should be bigger.
            # Note: grid_res should always be bigger than initial_set_radii, otherwise reachability behaves weirdly.
            "grid_res": 0.02,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
            "d_max": 0.0,
            # 'EVM_threshold': 0.3 # in m/s error when floating in forecasted vs sensed currents
            # 'fwd_back_buffer_in_seconds': 0.5,  # this is the time added to the earliest_to_reach as buffer for forward-backward
            "platform_dict": platform_dict,
        }
        super().__init__(problem, specific_settings)

        self.model = torch.load(
            f"{dirname(abspath(__file__))}/../../models/value_function_learning/baseline.pth"
        )

        self.true_cuurents = np.zeros((0, 4))

    def get_action(self, observation: ArenaObservation) -> PlatformAction:
        """
        Return action that goes in the direction of the target with full power.
        Args:
            observation: observation returned by the simulator
        Returns:
            PlatformAction dataclass
        """
        # 1. Update Planner
        action = super().get_action(observation)

        # 2. Append Measurements
        self.true_currents = np.append(
            self.true_cuurents,
            np.expand_dims(
                np.array(
                    [
                        observation.platform_state.lon.deg,
                        observation.platform_state.lat.deg,
                        observation.true_current_at_state.u,
                        observation.true_current_at_state.v,
                    ]
                ).squeeze(),
                axis=0,
            ),
            axis=0,
        )
        self.true_currents = self.true_currents[-TRUE_CURRENT_LENGTH:, :]
        print(self.true_cuurents.shape)

        return action

        # if self.true_currents.shape[0] >= 5:
        #     # 3. Get Grid from Model
        #     grid = get_value_function_grid(self, observation.platform_state.to_spatio_temporal_point(), TTR_MAP_IN_WIDTH, TTR_MAP_IN_WIDTH_DEG)
        #     x = get_x_train(grid, true_currents)
        #     improved_grid = self.model(x)
        #
        #     # 4. Get gradient and action
        #
        # else:
