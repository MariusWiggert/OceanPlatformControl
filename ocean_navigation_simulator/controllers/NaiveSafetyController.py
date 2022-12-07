import math
import numpy as np
from ocean_navigation_simulator.controllers.Controller import Controller
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.Platform import PlatformAction

from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
import xarray as xr


class NaiveSafetyController(Controller):
    """Naive safety controller. Steers in direction, furthest away from land."""

    gpus: float = 0.0

    def __init__(self, problem: NavigationProblem, specific_settings: dict):
        # TODO: currently ignored
        super().__init__(problem)
        # TODO: safetyproblem?
        # TODO: need config for init
        self.specific_settings = {
            "d_land_min": 40,
            "filepath_distance_map": "data/bathymetry/bathymetry_distance_res_0.083_0.083_max.nc",
        }  # km
        self.distance_map = xr.open_dataset(self.specific_settings["filepath_distance_map"])

    def find_bearing_to_max_distance_to_land(
        self, observation: ArenaObservation, num_directions_to_sample=16, radius_in_deg: float = 0.1
    ):
        bearing = np.linspace(0, 360, num_directions_to_sample, endpoint=False)
        # TODO: numpyify
        sampled_positions = [
            (
                observation.platform_state.lon.deg + np.cos(b) * radius_in_deg,
                observation.platform_state.lat.deg + np.sin(b) * radius_in_deg,
            )
            for b in np.deg2rad(bearing)
        ]

        min_d_to_land = [
            self.distance_map.interp(lon=p[0], lat=p[1])["distance"].data for p in sampled_positions
        ]
        bearing = bearing[np.argmax(min_d_to_land)]

        return bearing

    def get_action(self, observation: ArenaObservation) -> PlatformAction:

        # TODO: can land in local maximum :/
        distance_to_land_at_pos = self.distance_map.interp(
            lon=observation.platform_state.lon.deg, lat=observation.platform_state.lat.deg
        )["distance"].data
        # if distance_to_land_at_pos < self.specific_settings["d_land_min"]:

        bearing = self.find_bearing_to_max_distance_to_land(observation)
        dlon = np.cos(np.deg2rad(bearing))
        dlat = np.sin(np.deg2rad(bearing))
        mag = math.sqrt(dlon**2 + dlat**2)
        return PlatformAction.from_xy_propulsion(x_propulsion=dlon / mag, y_propulsion=dlat / mag)
