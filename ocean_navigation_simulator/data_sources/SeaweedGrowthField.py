from typing import Optional, Dict
from ocean_navigation_simulator.data_sources.DataField import DataField
from ocean_navigation_simulator.data_sources.SeaweedGrowth.SeaweedGrowthSource import SeaweedGrowthGEOMAR, SeaweedGrowthSource


class SeaweedGrowthField(DataField):
    """Class instantiating and holding the Seaweed Growth data sources, the forecast and hindcast sources.
  """
    hindcast_data_source: SeaweedGrowthSource = None
    forecast_data_source: SeaweedGrowthSource = None

    def __init__(self, casadi_cache_dict: Dict, hindcast_source_dict: Dict, forecast_source_dict: Optional[Dict] = None,
                 use_geographic_coordinate_system: Optional[bool] = True):
        """Initialize the source objects from the respective settings dicts.
        Args:
          casadi_cache_dict: containing the cache settings to use in the sources for caching of 3D data
                          e.g. {'deg_around_x_t': 2, 'time_around_x_t': 3600*24*12}

          forecast_source_dict and hindcast_source_dict
           Both are dicts with four keys:
             'field' the kind of field the should be created e.g. OceanCurrent or SolarIrradiance
             'source' in {opendap, hindcast_files, forecast_files}
             'subset_time_buffer_in_s' specifying the buffer applied to the time-interval when sub-setting an area
             'source_settings' dict that contains the specific settings required for the selected 'source'. See classes.
        """
        super().__init__(casadi_cache_dict, hindcast_source_dict, forecast_source_dict, use_geographic_coordinate_system)

    @staticmethod
    def instantiate_source_from_dict(source_dict: dict):
        """Helper function to instantiate an OceanCurrentSource object from the dict."""
        if source_dict['source'] == 'GEOMAR':
            return SeaweedGrowthGEOMAR(source_dict)
        else:
            raise NotImplementedError("Selected source {} in the SeaweedGrowthSource dict is not implemented.".format(
                source_dict['source']))






