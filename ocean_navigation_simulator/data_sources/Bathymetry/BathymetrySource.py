# TODO: what to import? is this necessary?
import logging
import os
from typing import Dict
from ocean_navigation_simulator.data_sources.DataSource import DataSource


class BathymetrySource:
    def __init__(self, source_config_dict: Dict):
        # TODO: what does this do?
        super().__init__(source_config_dict)  # TODO: this may be a problem since DataSource is 3d
        self.source_config_dict = source_config_dict
        self.logger = logging.getLoffer("areana.ocean_field.bathymetry_source")
        self.logger.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())

    # Realizing that Datasource is highly time dependent and doesn't make any sense for static data :/
