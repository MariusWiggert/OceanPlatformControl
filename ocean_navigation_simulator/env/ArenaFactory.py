import datetime
import string

from ocean_navigation_simulator.env.Arena import Arena
from ocean_navigation_simulator.env.PlatformState import PlatformState
from ocean_navigation_simulator.env.utils import units


class ArenaFactory:
    @staticmethod
    def create(scenario_name: string) -> Arena:
        if scenario_name == 'gulf_of_mexico':
            sim_cache_dict = {
                'deg_around_x_t': 0.2,
                'time_around_x_t': 3600 * 24  # * 3
            }
            platform_dict = {
                'battery_cap_in_wh': 400.0,
                'u_max_in_mps': 0.1,
                'motor_efficiency': 1.0,
                'solar_panel_size': 0.5,
                'solar_efficiency': 0.2,
                'drag_factor': 675,
                'dt_in_s': 600,
                'use_geographic_coordinate_system': True,
            }
            ocean_source_dict = {
                'field': 'OceanCurrents',
                'source': 'opendap',
                'source_settings': {
                    'service': 'copernicus',
                    'currents': 'total',
                    'USERNAME': 'mmariuswiggert',
                    'PASSWORD': 'tamku3-qetroR-guwneq',
                    'DATASET_ID': 'cmems_mod_glo_phy_anfc_merged-uv_PT1H-i',
                },
            }
            solar_source_dict = {
                'field': 'SolarIrradiance',
                'source': 'analytical_wo_caching',
                'source_settings': {
                    'boundary_buffers': [0.2, 0.2],
                    'x_domain': [-180, 180],
                    'y_domain': [-90, 90],
                    'temporal_domain': [datetime.datetime(2020, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
                                        datetime.datetime(2023, 1, 10, 0, 0, 0, tzinfo=datetime.timezone.utc)],
                    'spatial_resolution': 0.1,
                    'temporal_resolution': 3600,
                }
            }
            seaweed_source_dict = {
                'field': 'SeaweedGrowth',
                'source': 'GEOMAR',
                'source_settings': {
                    'filepath': './data/Nutrients/2021_monthly_nutrients_and_temp.nc'
                }
            }
            platform_state = PlatformState(
                date_time=datetime.datetime(year=2021, month=11, day=20, hour=12, minute=0, second=0, tzinfo=datetime.timezone.utc),
                lon=units.Distance(deg=-83.69599051807417),
                lat=units.Distance(deg=27.214803181574762),
                battery_charge=units.Energy(watt_hours=100),
                seaweed_mass=units.Mass(kg=0)
            )
        elif scenario_name == 'current_highway':
            sim_cache_dict = {
                'deg_around_x_t': 10,
                'time_around_x_t': 10  # * 3
            }
            platform_dict = {
                'battery_cap_in_wh': 400.0,
                'u_max_in_mps': 0.1,
                'motor_efficiency': 1.0,
                'solar_panel_size': 0.5,
                'solar_efficiency': 0.2,
                'drag_factor': 675,
                'dt_in_s': 0.1,
                'use_geographic_coordinate_system': False,
            }
            ocean_source_dict = {
                'field': 'OceanCurrents',
                'source': 'analytical',
                'source_settings': {
                    'name': 'FixedCurrentHighwayField',
                    'boundary_buffers': [0.2, 0.2],
                    'x_domain': [0, 10],
                    'y_domain': [0, 10],
                    'temporal_domain': [0, 10],
                    'spatial_resolution': 0.1,
                    'temporal_resolution': 1,
                    'y_range_highway': [4, 6],
                    'U_cur': 2,
                },
            }
            solar_source_dict = None
            seaweed_source_dict = None
            platform_state = PlatformState(
                date_time=datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc),
                lon=units.Distance(deg=0),
                lat=units.Distance(deg=0),
                battery_charge=units.Energy(joule=100),
                seaweed_mass=units.Mass(kg=0)
            )
        else:
            raise ValueError(f'{scenario_name} is not a defined arena.')

        arena = Arena(
            sim_cache_dict=sim_cache_dict,
            platform_dict=platform_dict,
            ocean_dict={'hindcast': ocean_source_dict, 'forecast': None},
            solar_dict={'hindcast': solar_source_dict, 'forecast': None},
            seaweed_dict={'hindcast': seaweed_source_dict, 'forecast': None}
        )
        observation = arena.reset(platform_state)

        return arena, platform_state, observation

