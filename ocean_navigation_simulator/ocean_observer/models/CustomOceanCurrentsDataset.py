import datetime
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch
import xarray as xr
from DateTime import DateTime
from torch.utils.data import Dataset

from ocean_navigation_simulator.environment.data_sources.OceanCurrentField import OceanCurrentField


class CustomOceanCurrentsDatasetSubgrid(Dataset):
    IDX_LON, IDX_LAT, IDX_TIME = 0, 1, 2
    MARGIN = 0.09
    GULF_MEXICO_WITHOUT_MARGIN = [[-97.84, -76.42], [18.08, 30]]
    # GULF_MEXICO = [[GULF_MEXICO_WITHOUT_MARGIN[0][0] + MARGIN, GULF_MEXICO_WITHOUT_MARGIN[0][1] - MARGIN],
    #                [GULF_MEXICO_WITHOUT_MARGIN[1][0] + MARGIN, GULF_MEXICO_WITHOUT_MARGIN[1][1] - MARGIN]]

    # Gulf mexico for tile of radius 1
    GULF_MEXICO = [[-95.362841, - 85.766062], [22.666666666666657, 26.333333333333343]]

    def __init__(self, ocean_dict: Dict[str, Any], start_date: DateTime, end_date: DateTime,
                 input_cell_size: Tuple[int, int, int], output_cell_size: Tuple[int, int, int],
                 cfg_dataset: dict,
                 spatial_resolution_forecast: Optional[float] = None,
                 temporal_resolution_forecast: Optional[float] = None,
                 spatial_resolution_hindcast: Optional[float] = None,
                 temporal_resolution_hindcast: Optional[float] = None,
                 dtype=torch.float64
                 ):
        self.ocean_field = OceanCurrentField(
            sim_cache_dict=None,
            hindcast_source_dict=ocean_dict['hindcast'],
            forecast_source_dict=ocean_dict['forecast'],
            use_geographic_coordinate_system=True
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
        self.time_horizon_input = datetime.timedelta(hours=cfg_dataset.get('time_horizon_input_h', 5))
        self.time_horizon_output = datetime.timedelta(hours=cfg_dataset.get('time_horizon_output_h', 1))
        if self.time_horizon_input.seconds / 3600 > 24 or self.time_horizon_output.seconds / 3600 > 24:
            raise Exception(
                "NOT supported time horizon input and output should be <= 24 hours or adapt dataset algorithm!!!!")

        self.dtype = dtype

        self.radius_lon = cfg_dataset.get('radius_lon', 2)  # in deg
        self.radius_lat = cfg_dataset.get('radius_lat', 2)  # in deg
        self.margin_forecast = cfg_dataset.get('margin_forecast', 0.2)
        self.margin_time = datetime.timedelta(hours=cfg_dataset.get("margin_time_in_h", 1))
        self.stride_tiles_dataset = cfg_dataset.get('stride_tiles_dataset', 0.5)
        self.stride_time_dataset = datetime.timedelta(hours=cfg_dataset.get("stride_time_dataset_h", 1))
        self.inputs, self.outputs = [], []
        self.duration_per_forecast_file = datetime.timedelta(hours=24)
        self.time_restart = datetime.time(hour=12, minute=30, second=1, tzinfo=datetime.timezone.utc)
        dims_sizes = [0, 0, 0]
        datetime_fc_start = self.start_date
        start_forecast = datetime.datetime.combine(datetime_fc_start,
                                                   self.time_restart) - self.duration_per_forecast_file
        self.whole_grid_fc = self.ocean_field.forecast_data_source \
            .get_data_over_area(*self.GULF_MEXICO,
                                [start_forecast,
                                 start_forecast + self.duration_per_forecast_file],
                                spatial_resolution=self.spatial_resolution_forecast,
                                temporal_resolution=self.temporal_resolution_forecast)

        while datetime_fc_start < self.end_date:
            lon = self.GULF_MEXICO[0][0] + self.radius_lon
            dims_sizes[0] = 0
            while lon + self.radius_lon + self.margin_forecast < self.GULF_MEXICO[0][1]:
                lat = self.GULF_MEXICO[1][0] + self.radius_lat + self.margin_forecast
                dims_sizes[1] = 0
                while lat + self.radius_lat + self.margin_forecast < self.GULF_MEXICO[1][1]:
                    self.__add_input_and_output_to_list(lon, lat, datetime_fc_start)
                    lat += self.stride_tiles_dataset
                    dims_sizes[1] += 1
                lon += self.stride_tiles_dataset
                dims_sizes[0] += 1
            while True:
                try:
                    self.__add_forecasts_to_xarray_if_necessary(datetime_fc_start)
                    datetime_fc_start += self.stride_time_dataset
                    break
                except ValueError:
                    datetime_fc_start += self.stride_time_dataset

            dims_sizes[2] += 1
        print("putting all inputs in memory")

        '''
        Instead load the whole area with min max time, good resolution space and time -> store that and get_idx just get the good slice.
        '''

        start_interval_dt = datetime.datetime.combine(datetime_fc_start,
                                                      self.time_restart) - self.duration_per_forecast_file
        self.__update_forecast_grid(start_interval_dt)
        self.whole_grid_fc = self.whole_grid_fc.drop("depth")
        self.whole_grid_hc = self.ocean_field.hindcast_data_source \
            .get_data_over_area(*self.GULF_MEXICO_WITHOUT_MARGIN,
                                [start_forecast, self.end_date + self.margin_time + self.time_horizon_output],
                                spatial_resolution=self.spatial_resolution_forecast,
                                temporal_resolution=self.temporal_resolution_forecast)
        # Interpolate the hindcast
        self.whole_grid_hc = self.whole_grid_hc.drop("depth").interp_like(self.whole_grid_fc, method='linear')

        self.merged = xr.merge([self.whole_grid_fc, self.whole_grid_hc.rename(water_u='hc_u', water_v='hc_v')])

        # Preload the data into memory
        self.whole_grid_fc.load()
        self.whole_grid_hc.load()
        self.merged.load()
        print("whole forecast grid dimensions:", dims_sizes)

    def __add_input_and_output_to_list(self, lon: float, lat: float, datetime_fc_start: DateTime):
        left_input, right_input = lon - self.radius_lon - self.margin_forecast, lon + self.radius_lon + self.margin_forecast
        bottom_input, top_input = lat - self.radius_lat - self.margin_forecast, lat + self.radius_lat + self.margin_forecast
        left_output, right_output = lon - self.radius_lon, lon + self.radius_lon
        bottom_output, top_output = lat - self.radius_lat, lat + self.radius_lat
        t1, t2_input, t2_output = datetime_fc_start, datetime_fc_start + self.time_horizon_input, datetime_fc_start + self.time_horizon_output
        # Check that both the input and output are in the data

        self.inputs.append(([left_input, right_input],
                            [bottom_input, top_input],
                            [t1, t2_input]))
        self.outputs.append(([left_output, right_output],
                             [bottom_output, top_output],
                             [t1, t2_output]))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # try:
        xy = self.__get_xarray_slice(self.inputs[idx], self.merged)
        # except ValueError as e:
        #     print(f"self.inputs[idx]: {self.inputs[idx]} not available. {e}")
        #     return None, None
        # X_xr = self.__get_xarray_slice(self.inputs[idx], self.whole_grid_fc)
        # X = X_xr.to_array().to_numpy()
        # y_xr = self.__get_xarray_slice(self.outputs[idx], self.whole_grid_hc)
        # y = y_xr.to_array().to_numpy()
        #
        # # Sanity check that X and y matches
        # assert (X_xr["time"][0:len(y_xr["time"])] == y_xr["time"]).all() and (X_xr["lon"] == y_xr["lon"]).all() and (
        #         X_xr["lat"] == y_xr["lat"]).all()
        lon, lat = self.input_shape[2], self.input_shape[3]
        # X = xy[["water_u", "water_v"]].to_array().to_numpy().astype('float32')[:, :, :lon, :lat]
        # y = xy[["hc_u", "hc_v"]].isel(time=[0]).to_array().to_numpy().astype('float32')[:, :, :lon, :lat]
        X = xy[[0, 1], :, :lon, :lat]
        y = xy[[2, 3], 0:1, :lon, :lat]

        if list(X.shape) != self.input_shape or list(y.shape) != self.output_shape or np.isnan(X).any() or np.isnan(
                y).any():
            # print([np.isnan(X_xr.isel(time=i).to_array().to_numpy()).sum() for i in range(5)],
            #       [np.isnan(y_xr.isel(time=i).to_array().to_numpy()).sum() for i in range(1)], X_xr.isel(
            #         time=0).to_array().to_numpy().size)
            return None
        # X, y = torch.tensor(X, dtype=self.dtype), torch.tensor(y, dtype=self.dtype)
        return X, y

    def __get_xarray_slice(self, field: list, grid: xr):
        lon, lat, time = field[0:3]
        lo, la, ti = np.logical_and(lon[0] < grid['lon'], grid['lon'] < lon[1]), \
                     np.logical_and(lat[0] < grid['lat'], grid['lat'] < lat[1]), \
                     np.logical_and(np.datetime64(time[0]) < grid['time'],
                                    grid['time'] < np.datetime64(time[1]))
        lo2, la2, ti2 = list(lo.to_numpy()), list(la.to_numpy()), list(ti.to_numpy())
        npg = grid.to_array().to_numpy()
        return npg[:, ti2][:, :, la2][..., lo2]
        # return grid.isel(lon=lo, lat=la, time=ti)

    def __add_forecasts_to_xarray_if_necessary(self, datetime_fc_start):
        # Add the forecasts to the merged xarray every 24 hours to get only the most recent forecast
        if datetime_fc_start <= datetime.datetime.combine(datetime_fc_start, self.time_restart) < (
                datetime_fc_start + self.stride_time_dataset):
            try:
                start_interval_dt = datetime.datetime.combine(datetime_fc_start,
                                                              self.time_restart) - self.duration_per_forecast_file

                self.__update_forecast_grid(start_interval_dt)
            except ValueError as e:
                print(e)

    def __update_forecast_grid(self, start_interval_dt):
        self.whole_grid_fc = self.ocean_field.forecast_data_source \
            .get_data_over_area(*self.GULF_MEXICO,
                                [start_interval_dt,
                                 start_interval_dt + self.duration_per_forecast_file],
                                spatial_resolution=self.spatial_resolution_forecast,
                                temporal_resolution=self.temporal_resolution_forecast).combine_first(
            self.whole_grid_fc)
