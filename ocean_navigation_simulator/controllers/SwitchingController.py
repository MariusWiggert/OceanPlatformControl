from ocean_navigation_simulator.controllers.Controller import Controller
from ocean_navigation_simulator.environment.Arena import Arena
from ocean_navigation_simulator.environment.Platform import PlatformAction
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.controllers.NaiveSafetyController import NaiveSafetyController
from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import (
    HJReach2DPlanner,
)
import xarray as xr


class SwitchingController(Controller):
    def __init__(self, problem: NavigationProblem, specific_settings: dict) -> None:
        super().__init__(problem)
        
        # TODO: Add config to config file
        # Create swich to create the controller that is wanted.
        hj_settings = {
            "replan_on_new_fmrc": True,
            "replan_every_X_seconds": False,
            "direction": "backward",
            "n_time_vector": 200,
            # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
            "deg_around_xt_xT_box": 1.0,  # area over which to run HJ_reachability
            "accuracy": "high",
            "artificial_dissipation_scheme": "local_local",
            "T_goal_in_seconds": 3600 * 24 * 5,
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
            # TODO: This is from the config. So we can perhaps pass the whole config and then pick what we need
            "platform_dict": {
                "battery_cap_in_wh": 400.0,
                "u_max_in_mps": 0.1,
                "motor_efficiency": 1.0,
                "solar_panel_size": 0.5,
                "solar_efficiency": 0.2,
                "drag_factor": 675.0,
                "dt_in_s": 600.0,
            },
        }

        self.specific_settings = specific_settings

        self.navigation_controller = HJReach2DPlanner(problem, specific_settings=hj_settings)
        self.safety_controller = NaiveSafetyController(
            problem=problem,
            specific_settings={
                "d_land_min": 40,
                "filepath_distance_map": "data/bathymetry/bathymetry_distance_res_0.083_0.083_max.nc",
            },
        )
        self.safety_status = False

    def safety_condition(self, observation: ArenaObservation, metric: str = "distance") -> bool:
        # If True, switch to safety
        if metric == "distance":
            distance_to_land_at_pos = self.distance_map.interp(
                lon=observation.platform_state.lon.deg, lat=observation.platform_state.lat.deg
            )["distance"].data
            return distance_to_land_at_pos < self.specific_settings["d_land_min"]
        elif metric == "distance_and_time_safe":
            # TODO: implement time counter
            raise NotImplementedError
        else:
            raise NotImplementedError

    def get_action(self, observation: ArenaObservation) -> PlatformAction:
        # Get the action depending on switching condition
        safety_status = self.safety_condition(observation)

        if safety != self.safety_status:
            self.logger.info(f"SwitchingController: Safety switched to {safety_status}")
            self.safety_status = safety_status

        if safety:
            self.safety_controller.get_action(observation)
        else:
            self.navigation_controller.get_action(observation)
