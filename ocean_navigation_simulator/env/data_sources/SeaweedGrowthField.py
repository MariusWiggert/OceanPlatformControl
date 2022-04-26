import datetime
from typing import List, NamedTuple, Sequence, Optional, Dict
import ocean_navigation_simulator.env.utils.units as units
from ocean_navigation_simulator.env.data_sources.DataField import DataField
from ocean_navigation_simulator.env.data_sources.SeaweedGrowth.SeaweedGrowthSource import SeaweedGrowthGEOMAR
import xarray as xr
from geopy.point import Point as GeoPoint


class SeaweedGrowthField(DataField):
    """Class instantiating and holding the Seaweed Growth data sources, the forecast and hindcast sources.
  """

    def __init__(self, sim_cache_dict: Dict, hindcast_source_dict: Dict, forecast_source_dict: Optional[Dict] = None):
        """Initialize the source objects from the respective settings dicts.
        Args:
          sim_cache_dict: containing the cache settings to use in the sources for caching of 3D data
                          e.g. {'deg_around_x_t': 2, 'time_around_x_t': 3600*24*12}

          forecast_source_dict and hindcast_source_dict
           Both are dicts with four keys:
             'field' the kind of field the should be created e.g. OceanCurrent or SolarIrradiance
             'source' in {opendap, hindcast_files, forecast_files}
             'subset_time_buffer_in_s' specifying the buffer applied to the time-interval when sub-setting an area
             'source_settings' dict that contains the specific settings required for the selected 'source'. See classes.
        """
        super().__init__(sim_cache_dict, hindcast_source_dict, forecast_source_dict)

    @staticmethod
    def instantiate_source_from_dict(source_dict: dict):
        """Helper function to instantiate an OceanCurrentSource object from the dict."""
        if source_dict['source'] == 'GEOMAR':
            return SeaweedGrowthGEOMAR(source_dict)
        else:
            raise NotImplementedError("Selected source {} in the SeaweedGrowthSource dict is not implemented.".format(
                source_dict['source']))





