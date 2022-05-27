import string
import yaml
from ocean_navigation_simulator.environment.Arena import Arena


class ArenaFactory:
    @staticmethod
    def create(scenario_name: string) -> Arena:
        with open(f'scenarios/{scenario_name}.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        return Arena(
            sim_cache_dict=config['sim_cache_dict'],
            platform_dict=config['platform_dict'],
            ocean_dict=config['ocean_dict'],
            use_geographic_coordinate_system=config['use_geographic_coordinate_system'],
            solar_dict=config['solar_dict'],
            seaweed_dict=config['seaweed_dict'],
        )