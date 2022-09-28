import datetime
from typing import Any, Dict, Optional, Tuple

import torch
from DateTime import DateTime
from torch.utils.data import Dataset

from ocean_navigation_simulator.data_sources.OceanCurrentField import (
    OceanCurrentField,
)


class CustomOceanCurrentsDataset(Dataset):
    IDX_LON, IDX_LAT, IDX_TIME = 0, 1, 2
    GULF_MEXICO = [[-97.84, -76.42], [18.12, 30]]

    def __init__(
        self,
        ocean_dict: Dict[str, Any],
        start_date: DateTime,
        end_date: DateTime,
        input_cell_size: Tuple[int, int, int],
        output_cell_size: Tuple[int, int, int],
        transform=None,
        target_transform=None,
        spatial_resolution_forecast: Optional[float] = None,
        temporal_resolution_forecast: Optional[float] = None,
        spatial_resolution_hindcast: Optional[float] = None,
        temporal_resolution_hindcast: Optional[float] = None,
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

    def __len__(self):
        return (self.end_date - self.start_date).days

    def __getitem__(self, idx):
        start = self.start_date + datetime.timedelta(days=idx)
        end_input = start + self.input_cell_size[self.IDX_TIME]
        # end_output = start + self.output_cell_size[self.IDX_TIME]
        input = self.ocean_field.forecast_data_source.get_data_over_area(
            *self.GULF_MEXICO,
            [start, end_input],
            spatial_resolution=self.spatial_resolution_forecast,
            temporal_resolution=self.temporal_resolution_forecast
        )
        output = self.ocean_field.hindcast_data_source.get_data_over_area(
            *self.GULF_MEXICO,
            [start, end_input],
            spatial_resolution=self.spatial_resolution_hindcast,
            temporal_resolution=self.temporal_resolution_hindcast
        )
        return torch.tensor(input.to_array().to_numpy()), torch.tensor(output.to_array().to_numpy())


class CustomOceanCurrentsDatasetSubgrid(Dataset):
    IDX_LON, IDX_LAT, IDX_TIME = 0, 1, 2
    MARGIN = 0.2
    GULF_MEXICO = [[-97.84 + MARGIN, -76.42 - MARGIN], [18.08 + MARGIN, 30 - MARGIN]]

    def __init__(
        self,
        ocean_dict: Dict[str, Any],
        start_date: DateTime,
        end_date: DateTime,
        input_cell_size: Tuple[int, int, int],
        output_cell_size: Tuple[int, int, int],
        transform=None,
        target_transform=None,
        spatial_resolution_forecast: Optional[float] = None,
        temporal_resolution_forecast: Optional[float] = None,
        spatial_resolution_hindcast: Optional[float] = None,
        temporal_resolution_hindcast: Optional[float] = None,
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
        self.time_horizon_nn = datetime.timedelta(hours=12)
        self.time_horizon_input = datetime.timedelta(days=5)

        radius_lon, radius_lat = 2, 2  # in deg
        margin_forecast = 0.2
        stride = 0.5
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
                    self.inputs.append(
                        (
                            [
                                lon - radius_lon - margin_forecast,
                                lon + radius_lon + margin_forecast,
                            ],
                            [
                                lat - radius_lat - margin_forecast,
                                lat + radius_lat + margin_forecast,
                            ],
                            [time, time + datetime.timedelta(days=5)],
                        )
                    )
                    self.outputs.append(
                        (
                            [lon - radius_lon, lon + radius_lon],
                            [lat - radius_lat, lat + radius_lat],
                            [time, time + self.time_horizon_nn],
                        )
                    )
                    lat += stride
                lon += stride
            time += datetime.timedelta(days=1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = (
            self.ocean_field.forecast_data_source.get_data_over_area(
                *self.inputs[idx],
                spatial_resolution=self.spatial_resolution_forecast,
                temporal_resolution=self.temporal_resolution_forecast
            )
            .to_array()
            .to_numpy()
        )

        output_grid = self.ocean_field.forecast_data_source.get_data_over_area(
            *self.outputs[idx],
            spatial_resolution=self.spatial_resolution_forecast,
            temporal_resolution=self.temporal_resolution_forecast
        )
        output = self.ocean_field.hindcast_data_source.get_data_over_area(
            *self.outputs[idx],
            spatial_resolution=self.spatial_resolution_hindcast,
            temporal_resolution=self.temporal_resolution_hindcast
        )

        # Todo: adapt in case to avoid interpolation
        output = output.interp_like(output_grid, method="linear").to_array().to_numpy()

        return (
            input[:, : int(self.time_horizon_input.total_seconds() // 3600)],
            output[:, 1 : int(1 + (self.time_horizon_nn.seconds // 3600))],
        )
