import datetime
from typing import List, NamedTuple, Sequence, Optional
from ocean_navigation_simulator.env.data_sources.DataField import DataField
from ocean_navigation_simulator.env.data_sources.OceanCurrentSource.OceanCurrentSource import OceanCurrentSourceXarray
from ocean_navigation_simulator.env.data_sources.OceanCurrentSource.OceanCurrentVector import OceanCurrentVector
from ocean_navigation_simulator.env.data_sources.OceanCurrentSource.OceanCurrentSource import HindcastFileSource, HindcastOpendapSource, ForecastFileSource
import ocean_navigation_simulator.env.data_sources.OceanCurrentSource.AnalyticalSource as AnalyticalSources
import xarray as xr
from geopy.point import Point as GeoPoint


class OceanCurrentField(DataField):
    """Class instantiating and holding the data sources, the forecast and hindcast current sources.
  """

    def __init__(self, hindcast_source_dict: dict, forecast_source_dict: Optional[dict] = None):
        """Initialize the source objects from the respective settings dicts.
        Args:
          forecast_source_dict and hindcast_source_dict
           Both are dicts with four keys:
             'field' the kind of field the should be created e.g. OceanCurrent or SolarIrradiance
             'source' in {opendap, hindcast_files, forecast_files}
             'subset_time_buffer_in_s' specifying the buffer applied to the time-interval when sub-setting an area
             'casadi_cache_settings': e.g. {'deg_around_x_t': 2, 'time_around_x_t': 3600*24*12} for caching of 3D data
             'source_settings' dict that contains the specific settings required for the selected 'source'. See classes.
        """
        super().__init__(hindcast_source_dict, forecast_source_dict)

    @staticmethod
    def instantiate_source_from_dict(source_dict: dict) -> OceanCurrentSourceXarray:
        """Helper function to instantiate an OceanCurrentSource object from the dict."""
        if source_dict['source'] == 'opendap':
            return HindcastOpendapSource(source_dict)
        elif source_dict['source'] == 'hindcast_files':
            return HindcastFileSource(source_dict)
        elif source_dict['source'] == 'forecast_files':
            return ForecastFileSource(source_dict)
        elif source_dict['source'] == 'analytical':
            specific_analytical_current = getattr(AnalyticalSources, source_dict['source_settings']['name'])
            return specific_analytical_current(source_dict)
        else:
            ValueError("Selected source {} in the OceanCurrentSource dict is not implemented.". format(source_dict['source']))





