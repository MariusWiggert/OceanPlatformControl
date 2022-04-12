import datetime as dt
import numpy as np

from ocean_navigation_simulator.env.Arena import Arena, ArenaObservation
from ocean_navigation_simulator.env.Platform import Platform, PlatformState
from ocean_navigation_simulator.env.controllers.naive_to_target_controller import NaiveToTargetController
from ocean_navigation_simulator.env.problem import Problem

from ocean_navigation_simulator.env.utils import units

platform_state = PlatformState(
    date_time=dt.datetime(year=2021, month=11, day=20, hour=12, minute=0, second=0, tzinfo=dt.timezone.utc),
    lon=units.Distance(deg=-83.69599051807417),
    lat=units.Distance(deg=27.214803181574762)
)
platform_dict = {
    'battery_cap_in_wh': 400.0,
    'u_max_in_mps': 0.1,
    'motor_efficiency': 1.0,
    'solar_panel_size': 0.5,
    'solar_efficiency': 0.2,
    'drag_factor': 675,
    'dt_in_s': 600,
}
ocean_source_dict = {
    'field': 'OceanCurrents',
    'subset_time_buffer_in_s': 4000,
    'casadi_cache_settings': {'deg_around_x_t': 2, 'time_around_x_t': 3600*24*1},
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
    'subset_time_buffer_in_s': 4000,
    'casadi_cache_settings': {'deg_around_x_t': 2, 'time_around_x_t': 3600*24*1},
    'source': 'analytical',
    'source_settings': {
        'boundary_buffers': [0.2, 0.2],
        'x_domain': [-180, 180],
        'y_domain': [-90, 90],
        'temporal_domain': [dt.datetime(2020, 1, 1, 0, 0, 0), dt.datetime(2023, 1, 10, 0, 0, 0)],
        'spatial_resolution': 0.1,
        'temporal_resolution': 3600,
    }
}

arena = Arena(
    platform_dict=platform_dict,
    ocean_dict={'forecast': None, 'hindcast': ocean_source_dict},
    solar_dict={'forecast': None, 'hindcast': solar_source_dict}
)
controller = NaiveToTargetController(problem=Problem(
    start_state = platform_state,
    end_region=PlatformState(
        date_time=platform_state.date_time,
        lon=units.Distance(deg=-83.99317002359827),
        lat=units.Distance(deg=27.32464464432537)
    )
))

observation = arena.reset(platform_state)

for i in range(6 * 40):
    action = controller.get_action(observation)
    observation = arena.step(action)

arena.do_nice_plot(x_T=np.array([controller.problem.end_region.lon.deg, controller.problem.end_region.lat.deg]))