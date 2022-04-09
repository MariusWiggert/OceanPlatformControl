import datetime
from typing import List, NamedTuple, Sequence, Optional
import ocean_navigation_simulator.utils.units as units
from ocean_navigation_simulator.env.data_sources.DataField import DataField
from ocean_navigation_simulator.env.data_sources.SolarIrradiance import *
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
    """Class instantiating and holding the data sources, the forecast and hindcast current sources.
  """

    @staticmethod
    def instantiate_source_from_dict(ocean_source_dict: dict) -> OceanCurrentSource:
        """Helper function to instantiate an OceanCurrentSource object from the dict."""
        if ocean_source_dict['source'] == 'opendap':
            return HindcastOpendapSource(ocean_source_dict)
        elif ocean_source_dict['source'] == 'hindcast_files':
            return HindcastFileSource(ocean_source_dict)
        elif ocean_source_dict['source'] == 'forecast_files':
            return ForecastFileSource(ocean_source_dict)
        else:
            ValueError("Selected source {} in the OceanCurrentSource dict is not implemented.". format(ocean_source_dict['source']))





