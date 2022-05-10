import datetime
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import xarray as xr

from ocean_navigation_simulator.env.Arena import Arena
from ocean_navigation_simulator.env.PlatformState import PlatformState
from ocean_navigation_simulator.env.data_sources.OceanCurrentSource.OceanCurrentVector import OceanCurrentVector
from ocean_navigation_simulator.env.utils import units
from ocean_navigation_simulator.env.utils.units import Velocity, Distance
from scripts.experiments.class_gp import OceanCurrentGP

# TODO: Change that such that the value is a function depending on the source of the hindcast
# 3600 = 1m/s * 24 * 60 * 60 =(Avg speed)* #seconds in the horizon
# _TIME_HORIZON_PREDICTIONS = datetime.timedelta(hours=24)
_TIME_HORIZON_PREDICTIONS = datetime.timedelta(seconds=24)
# _VELOCITY_FOR_AREA = Velocity(meters_per_second=1)
_VELOCITY_FOR_AREA = Velocity(meters_per_second=units.METERS_PER_DEG_LAT_LON / 12)
_RADIUS_AREA_AROUND_PLATFORM = _VELOCITY_FOR_AREA * _TIME_HORIZON_PREDICTIONS
print(
    f"dimension area around platform:{_RADIUS_AREA_AROUND_PLATFORM.m}m x {_RADIUS_AREA_AROUND_PLATFORM.m}m=" +
    f"{_RADIUS_AREA_AROUND_PLATFORM.m ** 2}m2")


def get_intervals_position_around_platform(platform_state: PlatformState, margin: Distance = Distance(m=0)) \
        -> Tuple[np.ndarray, np.ndarray]:
    return (np.asarray([(platform_state.lon - _RADIUS_AREA_AROUND_PLATFORM - margin).deg,
                        (platform_state.lon + _RADIUS_AREA_AROUND_PLATFORM + margin).deg]),
            np.asarray([(platform_state.lat - _RADIUS_AREA_AROUND_PLATFORM - margin).deg,
                        (platform_state.lat + _RADIUS_AREA_AROUND_PLATFORM + margin).deg]))


class Observer:
    def __init__(self, prediction_model: OceanCurrentGP, arena: Arena):
        self.model = prediction_model
        self.arena = arena

    def __get_area(self, platform_state: PlatformState, forecast: bool,
                   x_y_intervals: Optional[np.ndarray] = None) -> xr:
        fn = self.arena.ocean_field.get_forecast_area if forecast else self.arena.ocean_field.get_ground_truth_area
        if x_y_intervals is None:
            x_y_intervals = get_intervals_position_around_platform(platform_state)

        return fn(*x_y_intervals, [platform_state.date_time, platform_state.date_time + _TIME_HORIZON_PREDICTIONS])

    # def __get_forecasts_area(self, platform_state: PlatformState, x_y_intervals=None) -> xr:
    #     return self.__get_area(platform_state, True, x_y_intervals=x_y_intervals)

    def get_ground_truth_around_platform(self, platform_state: PlatformState,
                                         x_y_intervals: Optional[np.ndarray] = None) -> xr:
        return self.__get_area(platform_state, forecast=False, x_y_intervals=x_y_intervals)

    def get_forecast_around_platform(self, platform_state: PlatformState,
                                     x_y_intervals: Optional[np.ndarray] = None) -> xr:
        return self.__get_area(platform_state, forecast=True, x_y_intervals=x_y_intervals)

    def evaluate(self, platform_state: PlatformState, x_y_interval: Optional[np.ndarray] = None,
                 delta: datetime.timedelta = datetime.timedelta(seconds=0)) -> Tuple[xr.DataArray, xr.DataArray]:
        # 2) Query the whole grid around the platform if x_y_interval given
        area = self.get_forecast_around_platform(platform_state, x_y_intervals=x_y_interval)
        coords = np.array(np.meshgrid(area["lon"], area["lat"], pd.to_datetime(area["time"]))).transpose((3, 1, 2, 0))
        # Meshgrid shape before transpose: 3,lon,lat,time
        # Coords shape = time, lon, lat, 3=(#number dims meshgrid)
        locations = coords.reshape((-1, 3))
        locations[:, 2] = pd.to_datetime(locations[:, 2]) + delta
        mean, std = self.model.query_locations(locations)
        reshape_dims = (*coords.shape[:-1], 2)
        mean, std = mean.reshape(reshape_dims), std.reshape(reshape_dims)
        mean_xr = xr.DataArray(data=mean, dims=["time", "lon", "lat", "u_v"],
                               coords={"time": area["time"], "lon": area["lon"], "lat": area["lat"], "u_v": ["u", "v"]})
        std_xr = xr.DataArray(data=std, dims=["time", "lon", "lat", "u_v"],
                              coords={"time": area["time"], "lon": area["lon"], "lat": area["lat"], "u_v": ["u", "v"]})
        return mean_xr, std_xr

    def fit(self) -> None:
        self.model.fitting_GP()

    def observe(self, platform_state: PlatformState, difference_forecast_gt: OceanCurrentVector) -> None:
        self.model.observe(platform_state.lon.deg, platform_state.lat.deg, platform_state.date_time,
                           difference_forecast_gt)
