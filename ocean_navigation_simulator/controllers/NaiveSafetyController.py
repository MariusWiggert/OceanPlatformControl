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
    """Naive safety controller. Steers in direction furthest away from area.
    Can end up in local maxima."""

    gpus: float = 0.0

    def __init__(self, problem: NavigationProblem, specific_settings: dict) -> None:
        super().__init__(problem)
        self.specific_settings = specific_settings
        self.distance_map = dict()
        self.distance_map["bathymetry"] = xr.open_dataset(
            self.specific_settings["filepath_distance_map"]["bathymetry"]
        )
        self.distance_map["garbage"] = xr.open_dataset(
            self.specific_settings["filepath_distance_map"]["garbage"]
        )

    def find_bearing_to_max_distance_to_area(
        self,
        observation: ArenaObservation,
        area_type: str = "bathymetry",
        num_directions_to_sample=16,
        radius_in_deg: float = 0.1,
    ) -> float:
        """Find bearing to move platform furthest away from area.

        Args:
            observation (ArenaObservation): Current observation.
            area_type (str): Type of area to stay away from, e.g. "garbage", "bathymetry"
            num_directions_to_sample (int, optional): Number of directions to sample around platform. Defaults to 16.
            radius_in_deg (float, optional): Radius in degree to the samples. Defaults to 0.1.

        Returns:
            float: Bearing in degree to get the furthest away from area.
        """
        bearing = np.linspace(0, 360, num_directions_to_sample, endpoint=False)
        # TODO: numpyify
        sampled_positions = [
            (
                observation.platform_state.lon.deg + np.cos(b) * radius_in_deg,
                observation.platform_state.lat.deg + np.sin(b) * radius_in_deg,
            )
            for b in np.deg2rad(bearing)
        ]

        # TODO: Try if this map is available?
        min_d_to_area = [
            self.distance_map[area_type].interp(lon=p[0], lat=p[1])["distance"].data
            for p in sampled_positions
        ]
        bearing = bearing[np.argmax(min_d_to_area)]

        return bearing

    def get_action(
        self, observation: ArenaObservation, area_type: str = "bathymetry"
    ) -> PlatformAction:
        """Get action to actuate in the direction furthest away from area.

        Args:
            observation (ArenaObservation): Current Arena observation.
            area_type (str): Type of area to stay away from, e.g. "garbage", "bathymetry"

        Returns:
            PlatformAction: Which action to take.
        """
        bearing = self.find_bearing_to_max_distance_to_area(observation, area_type)
        dlon = np.cos(np.deg2rad(bearing))
        dlat = np.sin(np.deg2rad(bearing))
        mag = math.sqrt(dlon**2 + dlat**2)
        return PlatformAction.from_xy_propulsion(x_propulsion=dlon / mag, y_propulsion=dlat / mag)
