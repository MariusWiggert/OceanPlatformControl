import datetime
from typing import List, NamedTuple, Sequence, Optional
from ocean_navigation_simulator.env.data_sources.DataField import DataField
from ocean_navigation_simulator.env.data_sources.OceanCurrentSource.OceanCurrentSource import OceanCurrentSource
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
           Both are dicts with three keys:
             'source' in {opendap, hindcast_files, forecast_files}
             'subset_time_buffer_in_s' specifying the buffer applied to the time-interval when sub-setting an area
             'source_settings' dict that contains the specific settings required for the selected 'source'. See classes.
        """
        super().__init__(hindcast_source_dict, forecast_source_dict)

    @staticmethod
    def instantiate_source_from_dict(ocean_source_dict: dict) -> OceanCurrentSource:
        """Helper function to instantiate an OceanCurrentSource object from the dict."""
        if ocean_source_dict['source'] == 'opendap':
            return HindcastOpendapSource(ocean_source_dict)
        elif ocean_source_dict['source'] == 'hindcast_files':
            return HindcastFileSource(ocean_source_dict)
        elif ocean_source_dict['source'] == 'forecast_files':
            return ForecastFileSource(ocean_source_dict)
        elif ocean_source_dict['source'] == 'analytical':
            specific_analytical_current = getattr(AnalyticalSources, ocean_source_dict['source_settings']['name'])
            return specific_analytical_current(ocean_source_dict)
        else:
            ValueError("Selected source {} in the OceanCurrentSource dict is not implemented.". format(ocean_source_dict['source']))





