import logging
from typing import Dict, Optional

import ocean_navigation_simulator.data_sources.OceanCurrentSource.AnalyticalOceanCurrents as AnalyticalSources
from ocean_navigation_simulator.data_sources.DataField import DataField
from ocean_navigation_simulator.data_sources.OceanCurrentSource.OceanCurrentSource import (
    ForecastFileSource,
    HindcastFileSource,
    HindcastOpendapSource,
    LongTermAverageSource,
    OceanCurrentSource,
)


class OceanCurrentField(DataField):
    """Class instantiating and holding the data sources, the forecast and hindcast current sources."""

    hindcast_data_source: OceanCurrentSource = None
    forecast_data_source: OceanCurrentSource = None

    def __init__(
        self,
        casadi_cache_dict: Dict,
        hindcast_source_dict: Dict,
        forecast_source_dict: Optional[Dict] = None,
        use_geographic_coordinate_system: Optional[bool] = True,
    ):
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
        # initialize logger
        self.logger = logging.getLogger("arena.ocean_field")
        super().__init__(
            casadi_cache_dict,
            hindcast_source_dict,
            forecast_source_dict,
            use_geographic_coordinate_system,
        )

    @staticmethod
    def instantiate_source_from_dict(source_dict: Dict) -> OceanCurrentSource:
        """Helper function to instantiate an OceanCurrentSource object from the dict."""
        if source_dict["source"] == "opendap":
            return HindcastOpendapSource(source_dict)
        elif source_dict["source"] == "hindcast_files":
            return HindcastFileSource(source_dict)
        elif source_dict["source"] == "forecast_files":
            return ForecastFileSource(source_dict)
        elif source_dict["source"] == "longterm_average":
            return LongTermAverageSource(source_dict)
        elif source_dict["source"] == "analytical":
            specific_analytical_current = getattr(
                AnalyticalSources, source_dict["source_settings"]["name"]
            )
            return specific_analytical_current(source_dict)
        else:
            raise ValueError(
                "Selected source {} in the OceanCurrentSource dict is not implemented.".format(
                    source_dict["source"]
                )
            )
