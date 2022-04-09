import datetime
from typing import List, NamedTuple, Sequence, Optional
import ocean_navigation_simulator.env.utils.units as units
from ocean_navigation_simulator.env.data_sources.DataField import DataField
from ocean_navigation_simulator.env.data_sources.SolarIrradiance.SolarIrradianceSource import *
import xarray as xr
from geopy.point import Point as GeoPoint

# base = -377420400
# time = jnp.array([base])
# lat = jnp.arange(-89.5, 90.5, 1)
# lat = jnp.repeat(lat, 360) # repeat number = length of longitude range
# lon = jnp.arange(-179.5, 180.5, 1)
# lon = jnp.tile(lon, 180) # tile number = length of latitude range
# i_s = jnp.array(solar_rad(time, lat, lon)).reshape((180, 360))


class SolarIrradianceField(DataField):
    """Class instantiating and holding the Solar Irradiance data sources, the forecast and hindcast current sources.
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
    def instantiate_source_from_dict(source_dict: dict) -> SolarIrradianceSourceXarray:
        """Helper function to instantiate an OceanCurrentSource object from the dict."""
        if source_dict['source'] == 'analytical':
            return HindcastOpendapSource(source_dict)
        else:
            raise NotImplementedError("Selected source {} in the SolarIrradianceSource dict is not implemented.".format(
                source_dict['source']))






