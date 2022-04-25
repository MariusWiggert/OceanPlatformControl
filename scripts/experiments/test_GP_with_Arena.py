import datetime
import numpy as np

from ocean_navigation_simulator.env.Arena import Arena, ArenaObservation
from ocean_navigation_simulator.env.Platform import Platform, PlatformState
from ocean_navigation_simulator.env.controllers.unmotorized_controller import UnmotorizedController
from ocean_navigation_simulator.env.problem import Problem

from ocean_navigation_simulator.env.utils import units

_T_HORIZON_DAYS = 3
_DT_LAT, _DT_LONG = 0.5, 0.5
N_DATA_PTS = 100 # TODO: FINE TUNE THAT
#%%
initial_position = (-83.69599051807417, 27.214803181574762)
platform_state = PlatformState(
    date_time=datetime.datetime(year=2021, month=11, day=20, hour=12, minute=0, second=0, tzinfo=datetime.timezone.utc),
    lon=units.Distance(deg=initial_position[0]),
    lat=units.Distance(deg=initial_position[1]),
    seaweed_mass=units.Mass(kg=0),
    battery_charge=units.Energy(watt_hours=100)
)
#%%
sim_cache_dict = {'deg_around_x_t': 1, 'time_around_x_t': 3600 * 24 * 3}
platform_dict = {
    'battery_cap_in_wh': 400.0,
    'u_max_in_mps': 0.1,
    'motor_efficiency': 0, # We assume for now that the platform only follows the currents
    'solar_panel_size': 0.5,
    'solar_efficiency': 0.2,
    'drag_factor': 675,
    'dt_in_s': 600,
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

arena = Arena(
    sim_cache_dict=sim_cache_dict,
    platform_dict=platform_dict,
    ocean_dict={'hindcast': ocean_source_dict, 'forecast': None}
)

# %%
controller = UnmotorizedController(problem=Problem(
    start_state=platform_state,
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
