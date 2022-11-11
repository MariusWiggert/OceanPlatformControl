import datetime
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch
import xarray as xr
from DateTime import DateTime
from torch.utils.data import Dataset

from ocean_navigation_simulator.data_sources.OceanCurrentField import (
    OceanCurrentField,
)


# Class used to convert the Forecast/Hindcast files into input/target for the NN
# Used when the GP is not is not used


class CustomOceanCurrentsDatasetSubgrid(Dataset):
    IDX_LON, IDX_LAT, IDX_TIME = 0, 1, 2
    MARGIN = 0.09
    GULF_MEXICO_WITHOUT_MARGIN = [[-97.84, -76.42], [18.08, 30]]
    # Gulf mexico for tile of radius 1
    GULF_MEXICO = [[-95.362841, -85.766062], [22.666666666666657, 26.333333333333343]]

    def __init__(
        self,
        ocean_dict: Dict[str, Any],
        start_date: DateTime,
        end_date: DateTime,
        input_cell_size: Tuple[int, int, int],
        output_cell_size: Tuple[int, int, int],
        cfg_dataset: dict,
        spatial_resolution_forecast: Optional[float] = None,
        temporal_resolution_forecast: Optional[float] = None,
        spatial_resolution_hindcast: Optional[float] = None,
        temporal_resolution_hindcast: Optional[float] = None,
        dtype=torch.float64,
    ):
        self.ocean_field = OceanCurrentField(
            sim_cache_dict=None,
            hindcast_source_dict=ocean_dict["hindcast"],
            forecast_source_dict=ocean_dict["forecast"],
            use_geographic_coordinate_system=True,
        )
        self.input_cell_size = input_cell_size
        self.output_cell_size = output_cell_size
        self.start_date = start_date
        self.end_date = end_date
        self.spatial_resolution_forecast = spatial_resolution_forecast
        self.spatial_resolution_hindcast = spatial_resolution_hindcast
        self.temporal_resolution_forecast = temporal_resolution_forecast
        self.temporal_resolution_hindcast = temporal_resolution_hindcast
        self.input_shape = cfg_dataset["input_shape"]
        self.output_shape = cfg_dataset["output_shape"]
        self.time_horizon_input = datetime.timedelta(
            hours=cfg_dataset.get("time_horizon_input_h", 5)
        )
        self.time_horizon_output = datetime.timedelta(
            hours=cfg_dataset.get("time_horizon_output_h", 1)
        )
        if (
            self.time_horizon_input.seconds / 3600 > 24
            or self.time_horizon_output.seconds / 3600 > 24
        ):
            raise Exception(
                "NOT supported time horizon input and output should be <= 24 hours or adapt dataset algorithm!!!!"
            )

        self.dtype = dtype

        self.radius_lon = cfg_dataset.get("radius_lon", 2)  # in deg
        self.radius_lat = cfg_dataset.get("radius_lat", 2)  # in deg
        self.margin_fc_hc = cfg_dataset.get("margin_forecast", 0.2)
        self.margin_time = datetime.timedelta(hours=cfg_dataset.get("margin_time_in_h", 1))
        self.stride_tiles_dataset = cfg_dataset.get("stride_tiles_dataset", 0.5)
        self.stride_time_dataset = datetime.timedelta(
            hours=cfg_dataset.get("stride_time_dataset_h", 1)
        )
        self.inputs, self.outputs = [], []
        self.duration_per_forecast_file = datetime.timedelta(hours=24)
        self.time_restart = datetime.time(
            hour=12, minute=30, second=1, tzinfo=datetime.timezone.utc
        )
        sizes_of_lon_lat_time = [0, 0, 0]
        datetime_fc_start = self.start_date
        start_forecast = (
            datetime.datetime.combine(datetime_fc_start, self.time_restart)
            - self.duration_per_forecast_file
        )
        self.whole_grid_fc = self.ocean_field.forecast_data_source.get_data_over_area(
            *self.GULF_MEXICO,
            [start_forecast, start_forecast + self.duration_per_forecast_file],
            spatial_resolution=self.spatial_resolution_forecast,
            temporal_resolution=self.temporal_resolution_forecast
        ).isel(time=slice(0, 24 * self.duration_per_forecast_file.days))

        while datetime_fc_start < self.end_date:
            pos_lon = self.GULF_MEXICO[0][0] + self.radius_lon
            sizes_of_lon_lat_time[0] = 0
            while pos_lon + self.radius_lon + self.margin_fc_hc < self.GULF_MEXICO[0][1]:
                pos_lat = self.GULF_MEXICO[1][0] + self.radius_lat + self.margin_fc_hc
                sizes_of_lon_lat_time[1] = 0
                while pos_lat + self.radius_lat + self.margin_fc_hc < self.GULF_MEXICO[1][1]:
                    self.__add_input_and_output_to_list(pos_lon, pos_lat, datetime_fc_start)
                    pos_lat += self.stride_tiles_dataset
                    sizes_of_lon_lat_time[1] += 1
                pos_lon += self.stride_tiles_dataset
                sizes_of_lon_lat_time[0] += 1
            while True:
                try:
                    self.__add_forecasts_to_xarray_if_necessary(datetime_fc_start)
                    datetime_fc_start += self.stride_time_dataset
                    break
                except ValueError:
                    datetime_fc_start += self.stride_time_dataset

            sizes_of_lon_lat_time[2] += 1
        print("putting all inputs in memory")

        start_interval_dt = (
            datetime.datetime.combine(datetime_fc_start, self.time_restart)
            - self.duration_per_forecast_file
        )
        self.__update_forecast_grid(start_interval_dt)
        self.whole_grid_fc = self.whole_grid_fc.drop("depth")
        self.whole_grid_hc = self.ocean_field.hindcast_data_source.get_data_over_area(
            *self.GULF_MEXICO_WITHOUT_MARGIN,
            [
                start_forecast,
                self.end_date + self.margin_time + self.time_horizon_output,
            ],
            spatial_resolution=self.spatial_resolution_forecast,
            temporal_resolution=self.temporal_resolution_forecast
        )
        # Interpolate the hindcast
        self.whole_grid_hc = self.whole_grid_hc.drop("depth").interp_like(
            self.whole_grid_fc, method="linear"
        )

        self.merged = xr.merge(
            [
                self.whole_grid_fc,
                self.whole_grid_hc.rename(water_u="hc_u", water_v="hc_v"),
            ]
        )

        # Preload the data into memory
        # self.whole_grid_fc.load()
        # self.whole_grid_hc.load()
        self.merged.load()
        print("whole forecast grid dimensions:", sizes_of_lon_lat_time)

    def __add_input_and_output_to_list(self, lon: float, lat: float, datetime_fc_start: DateTime):
        left_input, right_input = (
            lon - self.radius_lon - self.margin_fc_hc,
            lon + self.radius_lon + self.margin_fc_hc,
        )
        bottom_input, top_input = (
            lat - self.radius_lat - self.margin_fc_hc,
            lat + self.radius_lat + self.margin_fc_hc,
        )
        left_output, right_output = (
            lon - self.radius_lon - self.margin_fc_hc,
            lon + self.radius_lon + self.margin_fc_hc,
        )
        bottom_output, top_output = (
            lat - self.radius_lat - self.margin_fc_hc,
            lat + self.radius_lat + self.margin_fc_hc,
        )
        t1, t2_input, t2_output = (
            datetime_fc_start,
            datetime_fc_start + self.time_horizon_input,
            datetime_fc_start + self.time_horizon_output,
        )

        self.inputs.append(([left_input, right_input], [bottom_input, top_input], [t1, t2_input]))
        self.outputs.append(
            ([left_output, right_output], [bottom_output, top_output], [t1, t2_output])
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # shape: time lat lon
        xy = self.__get_xarray_slice(self.inputs[idx], self.merged)

        lat, lon = self.input_shape[2], self.input_shape[3]
        t_input, t_ouput = self.input_shape[1], self.output_shape[1]
        i = 1 if xy.shape[-2] == lat + 2 else 0
        j = 1 if xy.shape[-1] == lon + 2 else 0
        assert xy.shape[-2] <= lat + i + 1 and xy.shape[-1] <= lon + j + 1

        X = xy[0:2, 0:t_input, i : lat + i, j : lon + j]
        y = xy[2:4, 0:t_ouput, i : lat + i, j : lon + j]

        # reShape as Current x time x lon x lat
        X = np.swapaxes(np.array(X), -1, -2)
        y = np.swapaxes(np.array(y), -1, -2)

        if (
            list(X.shape) != self.input_shape
            or list(y.shape) != self.output_shape
            or np.isnan(X).any()
            or np.isnan(y).any()
        ):
            return None
        return X, y

    def __get_xarray_slice(self, field: list, grid: xr):
        lon, lat, time = field[0:3]
        lo, la, ti = (
            np.logical_and(lon[0] < grid["lon"], grid["lon"] < lon[1]),
            np.logical_and(lat[0] < grid["lat"], grid["lat"] < lat[1]),
            np.logical_and(
                np.datetime64(time[0]) < grid["time"],
                grid["time"] < np.datetime64(time[1]),
            ),
        )
        lo2, la2, ti2 = list(lo.to_numpy()), list(la.to_numpy()), list(ti.to_numpy())
        grid_as_numpy = grid.to_array().to_numpy()
        # dims array returned: time, lat, lon
        return grid_as_numpy[:, ti2][:, :, la2][..., lo2]

    def __add_forecasts_to_xarray_if_necessary(self, datetime_fc_start):
        # Add the forecasts to the merged xarray every 24 hours to get only the most recent forecast
        if (
            datetime_fc_start
            <= datetime.datetime.combine(datetime_fc_start, self.time_restart)
            < (datetime_fc_start + self.stride_time_dataset)
        ):
            try:
                start_interval_dt = (
                    datetime.datetime.combine(datetime_fc_start, self.time_restart)
                    - self.duration_per_forecast_file
                )

                self.__update_forecast_grid(start_interval_dt)
            except ValueError as e:
                print(e)

    def __update_forecast_grid(self, start_interval_dt):
        self.whole_grid_fc = (
            self.ocean_field.forecast_data_source.get_data_over_area(
                *self.GULF_MEXICO,
                [
                    start_interval_dt,
                    start_interval_dt + self.duration_per_forecast_file,
                ],
                spatial_resolution=self.spatial_resolution_forecast,
                temporal_resolution=self.temporal_resolution_forecast
            )
            .isel(time=slice(0, self.duration_per_forecast_file.days * 24))
            .combine_first(self.whole_grid_fc)
        )
