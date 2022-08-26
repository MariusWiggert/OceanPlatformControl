import datetime
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch
from DateTime import DateTime
from torch.utils.data import Dataset

from ocean_navigation_simulator.environment.data_sources.OceanCurrentField import OceanCurrentField


# class CustomOceanCurrentsDataset(Dataset):
#     IDX_LON, IDX_LAT, IDX_TIME = 0, 1, 2
#     GULF_MEXICO = [[-97.84, -76.42], [18.12, 30]]
#
#     def __init__(self, ocean_dict: Dict[str, Any], start_date: DateTime, end_date: DateTime,
#                  input_cell_size: Tuple[int, int, int], output_cell_size: Tuple[int, int, int],
#                  transform=None, target_transform=None, spatial_resolution_forecast: Optional[float] = None,
#                  temporal_resolution_forecast: Optional[float] = None,
#                  spatial_resolution_hindcast: Optional[float] = None,
#                  temporal_resolution_hindcast: Optional[float] = None
#                  ):
#         self.ocean_field = OceanCurrentField(
#             sim_cache_dict=None,
#             hindcast_source_dict=ocean_dict['hindcast'],
#             forecast_source_dict=ocean_dict['forecast'],
#             use_geographic_coordinate_system=True
#         )
#         self.input_cell_size = input_cell_size
#         self.output_cell_size = output_cell_size
#         self.start_date = start_date
#         self.end_date = end_date
#         self.spatial_resolution_forecast = spatial_resolution_forecast
#         self.spatial_resolution_hindcast = spatial_resolution_hindcast
#         self.temporal_resolution_forecast = temporal_resolution_forecast
#         self.temporal_resolution_hindcast = temporal_resolution_hindcast
#
#     def __len__(self):
#         return (self.end_date - self.start_date).days
#
#     def __getitem__(self, idx):
#         start = self.start_date + datetime.timedelta(days=idx)
#         end_input = start + self.input_cell_size[self.IDX_TIME]
#         end_output = start + self.output_cell_size[self.IDX_TIME]
#         input = self.ocean_field.forecast_data_source.get_data_over_area(*self.GULF_MEXICO, [start, end_input],
#                                                                          spatial_resolution=self.spatial_resolution_forecast,
#                                                                          temporal_resolution=self.temporal_resolution_forecast)
#         output = self.ocean_field.hindcast_data_source.get_data_over_area(*self.GULF_MEXICO, [start, end_output],
#                                                                           spatial_resolution=self.spatial_resolution_hindcast,
#                                                                           temporal_resolution=self.temporal_resolution_hindcast)
#         return torch.tensor(input.to_array().to_numpy()), torch.tensor(output.to_array().to_numpy())


class CustomOceanCurrentsDatasetSubgrid(Dataset):
    IDX_LON, IDX_LAT, IDX_TIME = 0, 1, 2
    MARGIN = 0.5
    GULF_MEXICO_WITHOUT_MARGIN = [[-97.84, -76.42], [18.08, 30]]
    GULF_MEXICO = [[GULF_MEXICO_WITHOUT_MARGIN[0][0] + MARGIN, GULF_MEXICO_WITHOUT_MARGIN[0][1] - MARGIN],
                   [GULF_MEXICO_WITHOUT_MARGIN[1][0] + MARGIN, GULF_MEXICO_WITHOUT_MARGIN[1][1] - MARGIN]]

    def __init__(self, ocean_dict: Dict[str, Any], start_date: DateTime, end_date: DateTime,
                 input_cell_size: Tuple[int, int, int], output_cell_size: Tuple[int, int, int],
                 cfg_dataset: dict, transform=None, target_transform=None,
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
        self.dtype = dtype

        radius_lon = cfg_dataset.get('radius_lon', 2)  # in deg
        radius_lat = cfg_dataset.get('radius_lat', 2)  # in deg
        margin_forecast = cfg_dataset.get('margin_forecast', 0.2)
        margin_time = datetime.timedelta(hours=cfg_dataset.get("margin_time_in_h", 1))
        stride_tiles_dataset = cfg_dataset.get('stride_tiles_dataset', 0.5)
        stride_time_dataset = datetime.timedelta(hours=cfg_dataset.get("stride_time_dataset_h", 1))
        self.inputs, self.outputs = [], []
        dims_sizes = [0, 0, 0]
        datetime_fc_start = self.start_date
        time_restart = datetime.time(hour=12, minute=30, second=1, tzinfo=datetime.timezone.utc)
        duration_per_forecast_file = datetime.timedelta(hours=24)
        dims_sizes[2] = 0

        start_forecast = datetime.datetime.combine(datetime_fc_start, time_restart) - duration_per_forecast_file
        self.whole_grid_fc = self.ocean_field.forecast_data_source \
            .get_data_over_area(*self.GULF_MEXICO,
                                [start_forecast,
                                 start_forecast + duration_per_forecast_file],
                                spatial_resolution=self.spatial_resolution_forecast,
                                temporal_resolution=self.temporal_resolution_forecast)

        while datetime_fc_start < self.end_date:
            lon = self.GULF_MEXICO[0][0] + radius_lon
            dims_sizes[0] = 0
            while lon + radius_lon + margin_forecast < self.GULF_MEXICO[0][1]:
                lat = self.GULF_MEXICO[1][0] + radius_lat + margin_forecast
                dims_sizes[1] = 0
                while lat + radius_lat + margin_forecast < self.GULF_MEXICO[1][1]:
                    left_input, right_input = lon - radius_lon - margin_forecast, lon + radius_lon + margin_forecast
                    bottom_input, top_input = lat - radius_lat - margin_forecast, lat + radius_lat + margin_forecast
                    left_output, right_output = lon - radius_lon, lon + radius_lon
                    bottom_output, top_output = lat - radius_lat, lat + radius_lat
                    t1, t2_input, t2_output = datetime_fc_start, datetime_fc_start + self.time_horizon_input, datetime_fc_start + self.time_horizon_output
                    # Check that both the input and output are in the data

                    self.inputs.append(([left_input, right_input],
                                        [bottom_input, top_input],
                                        [t1, t2_input]))
                    self.outputs.append(([left_output, right_output],
                                         [bottom_output, top_output],
                                         [t1, t2_output]))
                    lat += stride_tiles_dataset
                    dims_sizes[1] += 1
                lon += stride_tiles_dataset
                dims_sizes[0] += 1
            if datetime_fc_start <= datetime.datetime.combine(datetime_fc_start, time_restart) < (
                    datetime_fc_start + stride_time_dataset):
                start_interval_dt = datetime.datetime.combine(datetime_fc_start,
                                                              time_restart) - duration_per_forecast_file
                self.whole_grid_fc = self.whole_grid_fc.merge(self.ocean_field.forecast_data_source \
                                                              .get_data_over_area(*self.GULF_MEXICO,
                                                                                  [start_interval_dt,
                                                                                   start_interval_dt + duration_per_forecast_file],
                                                                                  spatial_resolution=self.spatial_resolution_forecast,
                                                                                  temporal_resolution=self.temporal_resolution_forecast),
                                                              compat='override')
            datetime_fc_start += stride_time_dataset
            dims_sizes[2] += 1
        print("putting all inputs in memory")

        '''
        Instead load the whole area with min max time, good resolution space and time -> store that and get_idx just get the good slice.
        '''

        self.all_inputs = []
        start_interval_dt = datetime.datetime.combine(datetime_fc_start, time_restart) - duration_per_forecast_file
        print(start_interval_dt)
        self.whole_grid_fc = self.whole_grid_fc.merge(self.ocean_field.forecast_data_source \
                                                      .get_data_over_area(*self.GULF_MEXICO,
                                                                          [start_interval_dt,
                                                                           start_interval_dt + duration_per_forecast_file],
                                                                          spatial_resolution=self.spatial_resolution_forecast,
                                                                          temporal_resolution=self.temporal_resolution_forecast),
                                                      compat='override')
        self.whole_grid_hc = self.ocean_field.hindcast_data_source \
            .get_data_over_area(*self.GULF_MEXICO_WITHOUT_MARGIN,
                                [start_forecast, self.end_date + margin_time + self.time_horizon_output],
                                spatial_resolution=self.spatial_resolution_forecast,
                                temporal_resolution=self.temporal_resolution_forecast)
        # Interpolate the hindcast
        self.whole_grid_hc = self.whole_grid_hc.interp_like(self.whole_grid_fc, method='linear')

        # Preload the data into memory
        self.whole_grid_fc.load()
        self.whole_grid_hc.load()
        print("size dimensions:", dims_sizes)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        lon, lat, time = self.inputs[idx][0:3]

        lo, la, ti = np.logical_and(lon[0] < self.whole_grid_fc['lon'], self.whole_grid_fc['lon'] < lon[1]), \
                     np.logical_and(lat[0] < self.whole_grid_fc['lat'], self.whole_grid_fc['lat'] < lat[1]), \
                     np.logical_and(np.datetime64(time[0]) < self.whole_grid_fc['time'],
                                    self.whole_grid_fc['time'] < np.datetime64(time[1]))
        X = self.whole_grid_fc.isel(lon=lo, lat=la, time=ti).to_array().to_numpy()

        # X = self.whole_grid_fc.sel(lon=slice(*lon), lat=slice(*lat), time=slice(*time)).to_array().to_numpy()
        lon, lat, time = self.outputs[idx][0:3]
        lo, la, ti = np.logical_and(lon[0] < self.whole_grid_hc['lon'], self.whole_grid_hc['lon'] < lon[1]), \
                     np.logical_and(lat[0] < self.whole_grid_hc['lat'], self.whole_grid_hc['lat'] < lat[1]), \
                     np.logical_and(np.datetime64(time[0]) < self.whole_grid_hc['time'],
                                    self.whole_grid_hc['time'] < np.datetime64(time[1]))
        y = self.whole_grid_hc.isel(lon=lo, lat=la, time=ti).to_array().to_numpy()

        if list(X.shape) != self.input_shape or list(y.shape) != self.output_shape:
            print(time[0], time[1], lon[0], lat[0])
            return None
        X, y = torch.tensor(X, dtype=self.dtype), torch.tensor(y, dtype=self.dtype)
        X[torch.isnan(X)] = 0
        return X, y
