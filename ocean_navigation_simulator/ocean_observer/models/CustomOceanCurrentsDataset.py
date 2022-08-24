import datetime
from typing import Any, Dict, Tuple, Optional

import torch
from DateTime import DateTime
from torch.utils.data import Dataset

from ocean_navigation_simulator.environment.data_sources.OceanCurrentField import OceanCurrentField


class CustomOceanCurrentsDataset(Dataset):
    IDX_LON, IDX_LAT, IDX_TIME = 0, 1, 2
    GULF_MEXICO = [[-97.84, -76.42], [18.12, 30]]

    def __init__(self, ocean_dict: Dict[str, Any], start_date: DateTime, end_date: DateTime,
                 input_cell_size: Tuple[int, int, int], output_cell_size: Tuple[int, int, int],
                 transform=None, target_transform=None, spatial_resolution_forecast: Optional[float] = None,
                 temporal_resolution_forecast: Optional[float] = None,
                 spatial_resolution_hindcast: Optional[float] = None,
                 temporal_resolution_hindcast: Optional[float] = None
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

    def __len__(self):
        return (self.end_date - self.start_date).days

    def __getitem__(self, idx):
        start = self.start_date + datetime.timedelta(days=idx)
        end_input = start + self.input_cell_size[self.IDX_TIME]
        end_output = start + self.output_cell_size[self.IDX_TIME]
        input = self.ocean_field.forecast_data_source.get_data_over_area(*self.GULF_MEXICO, [start, end_input],
                                                                         spatial_resolution=self.spatial_resolution_forecast,
                                                                         temporal_resolution=self.temporal_resolution_forecast)
        output = self.ocean_field.hindcast_data_source.get_data_over_area(*self.GULF_MEXICO, [start, end_output],
                                                                          spatial_resolution=self.spatial_resolution_hindcast,
                                                                          temporal_resolution=self.temporal_resolution_hindcast)
        return torch.tensor(input.to_array().to_numpy()), torch.tensor(output.to_array().to_numpy())


class CustomOceanCurrentsDatasetSubgrid(Dataset):
    IDX_LON, IDX_LAT, IDX_TIME = 0, 1, 2
    MARGIN = 0.2
    GULF_MEXICO_WITHOUT_MARGIN = [[-97.84, -76.42], [18.08, 30]]
    GULF_MEXICO = [[GULF_MEXICO_WITHOUT_MARGIN[0][0] + MARGIN, GULF_MEXICO_WITHOUT_MARGIN[0][1] - MARGIN],
                   [GULF_MEXICO_WITHOUT_MARGIN[1][0] + MARGIN, GULF_MEXICO_WITHOUT_MARGIN[1][1] - MARGIN]]

    def __init__(self, ocean_dict: Dict[str, Any], start_date: DateTime, end_date: DateTime,
                 input_cell_size: Tuple[int, int, int], output_cell_size: Tuple[int, int, int],
                 transform=None, target_transform=None, spatial_resolution_forecast: Optional[float] = None,
                 temporal_resolution_forecast: Optional[float] = None,
                 spatial_resolution_hindcast: Optional[float] = None,
                 temporal_resolution_hindcast: Optional[float] = None,
                 cfg_database: Optional[dict] = {},
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
        self.time_horizon_input = datetime.timedelta(hours=cfg_database.get('time_horizon_input_h', 5))
        self.time_horizon_output = datetime.timedelta(hours=cfg_database.get('time_horizon_output_h', 1))
        self.dtype = dtype

        radius_lon = cfg_database.get('radius_lon', 2)  # in deg
        radius_lat = cfg_database.get('radius_lat', 2)  # in deg
        margin_forecast = cfg_database.get('margin_forecast', 0.2)
        stride_tiles_dataset = cfg_database.get('stride_tiles_dataset', 0.5)
        self.inputs, self.outputs = [], []
        time = self.start_date
        while time < self.end_date:
            lon = self.GULF_MEXICO[0][0] + radius_lon
            while lon + radius_lon + margin_forecast < self.GULF_MEXICO[0][1]:
                lat = self.GULF_MEXICO[1][0] + radius_lat + margin_forecast
                while lat + radius_lat + margin_forecast < self.GULF_MEXICO[1][1]:
                    # inputs.append(
                    #     self.ocean_field.forecast_data_source.get_data_over_area(
                    #         [lon - radius_lon - margin_forecast, lon + radius_lon + margin_forecast],
                    #         [lat - radius_lat - margin_forecast, lat + radius_lat + margin_forecast],
                    #         [self.start_date,
                    #          self.start_date + datetime.timedelta(days=5)], spatial_resolution=spatial_resolution_forecast,
                    #         temporal_resolution=temporal_resolution_forecast))
                    # outputs.append(
                    #     self.ocean_field.hindcast_data_source.get_data_over_area(
                    #         [lon - radius_lon, lon + radius_lon],
                    #         [lat - radius_lat, lat + radius_lat],
                    #         [self.start_date, self.start_date + datetime.timedelta(days=1)],
                    #         spatial_resolution=spatial_resolution_hindcast,
                    #         temporal_resolution=temporal_resolution_hindcast
                    #     )
                    # )
                    self.inputs.append(([lon - radius_lon - margin_forecast, lon + radius_lon + margin_forecast],
                                        [lat - radius_lat - margin_forecast, lat + radius_lat + margin_forecast],
                                        [time, time + self.time_horizon_input]))
                    self.outputs.append(([lon - radius_lon, lon + radius_lon],
                                         [lat - radius_lat, lat + radius_lat],
                                         [time, time + self.time_horizon_output]))
                    lat += stride_tiles_dataset
                lon += stride_tiles_dataset
            time += datetime.timedelta(days=1)
        print("putting all inputs in memory")

        '''
        Instead load the whole area with min max time, good resolution space and time -> store that and get_idx just get the good slice.
        '''

        self.all_inputs = []
        self.whole_grid_fc = self.ocean_field.forecast_data_source \
            .get_data_over_area(*self.GULF_MEXICO,
                                [self.start_date, self.end_date],
                                spatial_resolution=self.spatial_resolution_forecast,
                                temporal_resolution=self.temporal_resolution_forecast)
        self.whole_grid_hc = self.ocean_field.hindcast_data_source \
            .get_data_over_area(*self.GULF_MEXICO_WITHOUT_MARGIN,
                                [self.start_date, self.end_date],
                                spatial_resolution=self.spatial_resolution_forecast,
                                temporal_resolution=self.temporal_resolution_forecast)
        # Interpolate the hc
        self.whole_grid_hc = self.whole_grid_hc.interp_like(self.whole_grid_fc, method='linear')

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        lon, lat, time = self.inputs[idx][0:3]
        input = self.whole_grid_fc.sel(lon=slice(*lon), lat=slice(*lat), time=slice(*time)).to_array().to_numpy()
        lon, lat, time = self.outputs[idx][0:3]
        output = self.whole_grid_hc.sel(lon=slice(*lon), lat=slice(*lat), time=slice(*time)).to_array().to_numpy()
        input, output = torch.tensor(input, dtype=self.dtype), torch.tensor(output, dtype=self.dtype)
        input[torch.isnan(input)] = 0
        return input, output
