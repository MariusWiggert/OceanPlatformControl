import datetime as dt

import numpy as np
import xarray as xr
from tqdm import tqdm
import time

from ocean_navigation_simulator import Problem
from ocean_navigation_simulator.env.data_sources.OceanCurrentField import OceanCurrentField
from ocean_navigation_simulator.env.data_sources.SolarIrradianceField import SolarIrradianceField
from ocean_navigation_simulator.env.Platform import Platform, PlatformState, PlatformAction
from ocean_navigation_simulator.env.utils import units
from ocean_navigation_simulator.utils import plotting_utils, simulation_utils

script_start = time.time()

t_0 = dt.datetime(year=2021, month=11, day=23, hour=12, minute=10, second=10, tzinfo=dt.timezone.utc)
horizon_in_s = 3600 * 24 * 5

##### Initialize Datasource #####
start = time.time()
ocean_source_dict = {
    'field': 'OceanCurrents',
    'subset_time_buffer_in_s': 4000,
    'casadi_cache_settings': {'deg_around_x_t': 2, 'time_around_x_t': 3600*24*10},
    'source': 'opendap',
    'source_settings': {
        'service': 'copernicus',
        'currents': 'total',
        'USERNAME': 'mmariuswiggert',
        'PASSWORD': 'tamku3-qetroR-guwneq',
        'DATASET_ID': 'cmems_mod_glo_phy_anfc_merged-uv_PT1H-i',
    },
}
ocean_field = OceanCurrentField(hindcast_source_dict=ocean_source_dict)
solar_source_dict = {
    'field': 'SolarIrradiance',
    'subset_time_buffer_in_s': 4000,
    'casadi_cache_settings': {'deg_around_x_t': 2, 'time_around_x_t': 3600*24*10},
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
solar_field = SolarIrradianceField(hindcast_source_dict=solar_source_dict)
print(f'Init Sources: {time.time()-start:.2f}s')

##### Initialize Platform #####
start = time.time()
init_state = PlatformState(
    date_time=t_0,
    lon=units.Distance(deg=-81.5),
    lat=units.Distance(deg=23.5)
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
platform = Platform(state=init_state, platform_dict=platform_dict, ocean_source=ocean_field.hindcast_data_source, solar_source=solar_field.hindcast_data_source)
print(f'Init Platform: {time.time()-start:.2f}s')

steps = int(horizon_in_s / platform_dict['dt_in_s'])
trajectory = np.zeros((steps+1, 2))
action_sequence = np.zeros((steps, 2))
trajectory[0] = np.array([init_state.lon.deg, init_state.lat.deg])

##### Run Simulation #####
for step in tqdm(range(steps)):
    action = PlatformAction(magnitude=1,direction=3.14/2)
    state = platform.simulate_step(action=action)
    trajectory[step+1] = np.array([state.lon.deg, state.lat.deg])
    action_sequence[step] = np.array([action.magnitude, action.direction])

##### Plotting using old Utils #####
start = time.time()
data_store = simulation_utils.copernicusmarine_datastore('cmems_mod_glo_phy_anfc_merged-uv_PT1H-i','mmariuswiggert','tamku3-qetroR-guwneq')
DS_currents = xr.open_dataset(data_store)[['uo', 'vo']].isel(depth=0)
file_dicts = {'data_source_type': 'cop_opendap', 'content': DS_currents, 'grid_dict': Problem.derive_grid_dict_from_xarray(DS_currents)}
plotting_utils.plot_2D_traj_over_currents(
    x_traj=trajectory.T,
    deg_around_x0_xT_box=0.5,
    time=t_0.timestamp(),
    file_dicts=file_dicts,
    ctrl_seq=action_sequence.T,
    u_max=0.1
)
print(f'Plotting: {time.time()-start:.2f}s')

print(f'Total Script: {time.time()-script_start:.2f}s')