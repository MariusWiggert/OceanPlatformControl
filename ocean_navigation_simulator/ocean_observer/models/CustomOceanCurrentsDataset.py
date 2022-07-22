from typing import Any, Dict, Tuple

from DateTime import DateTime
from torch.utils.data import Dataset

from ocean_navigation_simulator.environment.data_sources.OceanCurrentField import OceanCurrentField


class CustomOceanCurrentsDataset(Dataset):
    def __init__(self, ocean_dict: Dict[str, Any], start_date: DateTime, end_date: DateTime,
                 input_cell_size: Tuple[int, int, int], output_cell_size: Tuple[int, int, int],
                 transform=None, target_transform=None):
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

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return None, None
