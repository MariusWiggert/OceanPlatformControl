import string
import yaml
from pathlib import Path
from os.path import dirname, abspath
import os

from ocean_navigation_simulator.env.Arena import Arena


class ArenaFactory:
    @staticmethod
    def create(scenario_name: string) -> Arena:
        print('test')
        print(__file__)
        print(Path(__file__).parent.resolve())
        print(Path(__file__).parent)
        print(dirname(abspath(__file__)))

        print(Path().resolve())
        print(abspath(os.getcwd()))

        with open(f'{dirname(abspath(__file__))}/scenarios/{scenario_name}.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        return Arena(
            sim_cache_dict=config['sim_cache_dict'],
            platform_dict=config['platform_dict'],
            ocean_dict=config['ocean_dict'],
            use_geographic_coordinate_system=config['use_geographic_coordinate_system'],
            solar_dict=config['solar_dict'],
            seaweed_dict=config['seaweed_dict'],
            spatial_boundary=config['spatial_boundary']
        )