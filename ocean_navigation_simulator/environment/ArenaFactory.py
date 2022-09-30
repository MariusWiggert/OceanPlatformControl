import string

import yaml

from ocean_navigation_simulator.environment.Arena import Arena


class ArenaFactory:
    @staticmethod
    def create(scenario_name: string, folder_scenario: string = None) -> Arena:
        path = (f'scenario/' if folder_scenario is None else f'{folder_scenario}/') + f'{scenario_name}.yaml'
        with open(path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        return Arena(
            sim_cache_dict=config['sim_cache_dict'],
            platform_dict=config['platform_dict'],
            ocean_dict=config['ocean_dict'],
            use_geographic_coordinate_system=config['use_geographic_coordinate_system'],
            solar_dict=config['solar_dict'],
            seaweed_dict=config['seaweed_dict'],
        )
