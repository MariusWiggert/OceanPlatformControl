import logging
import os
from typing import Dict, Optional

from ocean_navigation_simulator.data_sources.DataField import DataField
from ocean_navigation_simulator.data_sources.Bathymetry.BathymetrySource import BathymetrySource


class BathymetryField(DataField):
    """Class instantiating and holding the Bathymetry data source"""

    data_source: BathymetrySource = None

    def __init__(
        self,
        casadi_cache_dict: Dict,
        data_source_dict: Dict,
        use_geographic_coordinate_system: Optional[bool] = True,
    ):
        # TODO: problem if casadi_cache_dict is 2D?
        # TODO: finish docstring for data_source dict after writing it
        """Initialize the source objects from the respective settings dicts.

        Args:
            casadi_cache_dict (Dict): containing the cache settings to use in the sources for caching of 2D data
                          e.g. {'deg_around_x_t': 2}
            data_source_dict (Dict): _description_
            use_geographic_coordinate_system (Optional[bool], optional): Define which geographic coordinate system to use. Defaults to True.
        """
        # Initialize logger
        self.logger = logging.getLogger("arena.bathymetry_field")
        self.logger.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())
        # TODO: basically passing the data_source_dict as a hindcast_dict
        super().__init__(
            casadi_cache_dict=casadi_cache_dict,
            forecast_source_dict=data_source_dict,
            use_geographic_coordinate_system=use_geographic_coordinate_system,
        )

    @staticmethod
    def instantiate_source_from_dict(source_dict: dict):
        """Helper fnction to instantiate an BathymetrySource object from the dict."""
        if source_dict["source"] == "gebco":
            return BathymetrySource(source_dict)
        else:
            raise NotImplementedError(
                f"Selected source {source_dict['source']} in the BathymetrySource dict is not implemented."
            )
