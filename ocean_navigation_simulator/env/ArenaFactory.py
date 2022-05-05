import datetime
import string

import yaml

from ocean_navigation_simulator.env.Arena import Arena, ArenaObservation
from ocean_navigation_simulator.env.PlatformState import PlatformState, SpatialPoint
from ocean_navigation_simulator.env.utils import units


class ArenaFactory:
    @staticmethod
    def create(scenario_name: string) -> tuple[Arena, PlatformState, ArenaObservation, SpatialPoint]:
        with open(f'ocean_navigation_simulator/env/scenarios/{scenario_name}.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        arena = Arena(
            sim_cache_dict=config['sim_cache_dict'],
            platform_dict=config['platform_dict'],
            ocean_dict=config['ocean_dict'],
            solar_dict=config['solar_dict'],
            seaweed_dict=config['seaweed_dict'],
        )
        platform_state = PlatformState(
            lon=units.Distance(deg=config['platform_state']['lon']),
            lat=units.Distance(deg=config['platform_state']['lat']),
            date_time=config['platform_state']['date_time'],
            battery_charge=units.Energy(joule=config['platform_state']['battery_charge']),
            seaweed_mass=units.Mass(kg=config['platform_state']['seaweed_mass']),
        )
        observation = arena.reset(platform_state)
        end_region = SpatialPoint(
            lon=units.Distance(deg=config['end_region']['lon']),
            lat=units.Distance(deg=config['end_region']['lat']),
        )

        return arena, platform_state, observation, end_region

