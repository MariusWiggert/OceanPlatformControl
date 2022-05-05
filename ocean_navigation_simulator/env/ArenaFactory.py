import datetime
import string
from typing import Tuple

import yaml

from ocean_navigation_simulator.env.Arena import Arena, ArenaObservation
from ocean_navigation_simulator.env.PlatformState import PlatformState, SpatialPoint
from ocean_navigation_simulator.env.utils import units


class ArenaFactory:
    @staticmethod
    def create(scenario_name: string) -> Arena:
        with open(f'ocean_navigation_simulator/env/scenarios/{scenario_name}.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        return Arena(
            sim_cache_dict=config['sim_cache_dict'],
            platform_dict=config['platform_dict'],
            ocean_dict=config['ocean_dict'],
            use_geographic_coordinate_system=config['use_geographic_coordinate_system'],
            solar_dict=config['solar_dict'],
            seaweed_dict=config['seaweed_dict'],
        )