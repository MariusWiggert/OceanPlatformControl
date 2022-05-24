import datetime
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import xarray as xr

from ocean_navigation_simulator.env.Arena import Arena
from ocean_navigation_simulator.env.PlatformState import PlatformState
from ocean_navigation_simulator.env.data_sources.OceanCurrentSource.OceanCurrentVector import OceanCurrentVector
from ocean_navigation_simulator.env.models.OceanCurrentGP import OceanCurrentGP_old
from ocean_navigation_simulator.env.utils import units
from ocean_navigation_simulator.env.utils.units import Velocity, Distance


class Observer:
    def __init__(self, prediction_model: OceanCurrentGP_old, arena: Arena, config: Dict[str, Any],
                 general_config_file: Dict[str, Any] = {}):
        self.model = prediction_model
        self.arena = arena

        # TODO: Change that such that the value is a function depending on the source of the hindcast
        # 3600 = 1m/s * 24 * 60 * 60 =(Avg speed)* #seconds in the horizon
        # _TIME_HORIZON_PREDICTIONS = datetime.timedelta(hours=24)
        self._TIME_HORIZON_PREDICTIONS = datetime.timedelta(
            seconds=config["life_span_observations_in_sec"])

        # velocity_for_area = Velocity(meters_per_second=1 if general_config_file.get("use_real_data", True) else (
        #        units.METERS_PER_DEG_LAT_LON / 12))
        # todo: Remove because too small (only for testing)
        velocity_for_area = Velocity(meters_per_second=.06 if general_config_file.get("use_real_data", True) else (
                units.METERS_PER_DEG_LAT_LON / 12))
        self._RADIUS_AREA_AROUND_PLATFORM = velocity_for_area * self._TIME_HORIZON_PREDICTIONS
        print(
            f"dimension area around platform:{self._RADIUS_AREA_AROUND_PLATFORM.m}m x {self._RADIUS_AREA_AROUND_PLATFORM.m}m=" +
            f"dimension area around platform:{self._RADIUS_AREA_AROUND_PLATFORM.deg}deg x {self._RADIUS_AREA_AROUND_PLATFORM.deg}deg=" +
            f"{self._RADIUS_AREA_AROUND_PLATFORM.m ** 2}m2")

    def get_area_around_platform(self, platform_state: PlatformState, margin: Distance = Distance(m=0)) \
            -> Tuple[np.ndarray, np.ndarray]:
        return (np.asarray([(platform_state.lon - self._RADIUS_AREA_AROUND_PLATFORM - margin).deg,
                            (platform_state.lon + self._RADIUS_AREA_AROUND_PLATFORM + margin).deg]),
                np.asarray([(platform_state.lat - self._RADIUS_AREA_AROUND_PLATFORM - margin).deg,
                            (platform_state.lat + self._RADIUS_AREA_AROUND_PLATFORM + margin).deg]))

    def __get_area(self, platform_state: PlatformState, forecast: bool,
                   x_y_intervals: Optional[np.ndarray] = None, temporal_resolution: Optional[float] = None) -> xr:
        fn = self.arena.ocean_field.get_forecast_area if forecast else self.arena.ocean_field.get_ground_truth_area
        if x_y_intervals is None:
            x_y_intervals = self.get_area_around_platform(platform_state)
        return fn(*x_y_intervals, [platform_state.date_time, platform_state.date_time + self._TIME_HORIZON_PREDICTIONS],
                  temporal_resolution=temporal_resolution)

    def get_ground_truth_around_platform(self, platform_state: PlatformState,
                                         x_y_intervals: Optional[np.ndarray] = None,
                                         temporal_resolution: Optional[float] = None) -> xr:
        return self.__get_area(platform_state, forecast=False, x_y_intervals=x_y_intervals,
                               temporal_resolution=temporal_resolution)

    def get_forecast_around_platform(self, platform_state: PlatformState,
                                     x_y_intervals: Optional[np.ndarray] = None,
                                     temporal_resolution: Optional[float] = None) -> xr:
        return self.__get_area(platform_state, forecast=True, x_y_intervals=x_y_intervals,
                               temporal_resolution=temporal_resolution)

    def evaluate(self, platform_state: PlatformState, x_y_interval: Optional[np.ndarray] = None,
                 temporal_resolution: Optional[float] = None,
                 delta: datetime.timedelta = datetime.timedelta(seconds=0)) -> Tuple[xr.Dataset, xr.Dataset]:
        # Query the whole grid around the platform if x_y_interval given
        area = self.get_forecast_around_platform(platform_state, x_y_intervals=x_y_interval,
                                                 temporal_resolution=temporal_resolution)
        coords = np.array(np.meshgrid(area["lon"], area["lat"], pd.to_datetime(area["time"]))).transpose((3, 1, 2, 0))
        # Meshgrid shape before transpose: 3,lon,lat,time
        # Coords shape = time, lon, lat, 3=(#number dims meshgrid)
        locations = coords.reshape((-1, 3))
        locations[:, 2] = pd.to_datetime(locations[:, 2]) + delta
        mean, std = self.model.query_locations(locations)
        reshape_dims = (*coords.shape[:-1], 2)
        mean, std = mean.reshape(reshape_dims), std.reshape(reshape_dims)
        mean_xr = xr.Dataset(
            {"water_u": (["time", "lat", "lon"], mean[..., 0]),
             "water_v": (["time", "lat", "lon"], mean[..., 1])},
            coords={
                "lon": area['lon'],
                "lat": area['lat'],
                "time": area["time"]}
        )
        std_xr = xr.Dataset(
            {"water_u": (["time", "lat", "lon"], std[..., 0]),
             "water_v": (["time", "lat", "lon"], std[..., 1])},
            coords={
                "lon": area['lon'],
                "lat": area['lat'],
                "time": area["time"]}
        )
        return mean_xr, std_xr

    def fit(self) -> None:
        self.model.fit()

    def observe(self, platform_state: PlatformState, difference_forecast_gt: OceanCurrentVector) -> None:
        self.model.observe(platform_state.to_spatio_temporal_point(),
                           difference_forecast_gt)
