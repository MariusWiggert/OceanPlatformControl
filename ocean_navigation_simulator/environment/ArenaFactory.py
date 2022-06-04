import string
import yaml
from os.path import dirname, abspath

from ocean_navigation_simulator.environment.Arena import Arena

class ArenaFactory:
    @staticmethod
    def create(scenario_name: string = None, file: string = None) -> Arena:
        if scenario_name:
            with open(f'{dirname(abspath(__file__))}/../../scenarios/{scenario_name}.yaml') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        elif file:
            with open(file) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise ValueError('Specify scenario name or file.')

        return Arena(
            sim_cache_dict=config['sim_cache_dict'],
            platform_dict=config['platform_dict'],
            ocean_dict=config['ocean_dict'],
            use_geographic_coordinate_system=config['use_geographic_coordinate_system'],
            solar_dict=config['solar_dict'],
            seaweed_dict=config['seaweed_dict'],
            spatial_boundary=config['spatial_boundary']
        )