import datetime

import numpy as np
import xarray as xr
from tqdm import tqdm
import time

from ocean_navigation_simulator.data_sources.OceanCurrentSource.OceanCurrentSource import ForecastFileSource, HindcastFileSource, HindcastOpendapSource
from ocean_navigation_simulator.data_sources.OceanCurrentFields import OceanCurrentField
from ocean_navigation_simulator.env.platform import Platform, PlatformState, PlatformSpecs
from ocean_navigation_simulator.utils import plotting_utils, simulation_utils, solar_rad, units
from ocean_navigation_simulator.problem import Problem
from ocean_navigation_simulator import utils

script_start = time.time()

t_0 = datetime.datetime(year=2021, month=11, day=23, hour=12, minute=10, second=10, tzinfo=datetime.timezone.utc)
horizon = datetime.timedelta(days=5)
stride = datetime.timedelta(minutes=1)
steps = int(horizon / stride)

##### Initialize Datasource #####
start = time.time()
source = HindcastOpendapSource(
    source_type='string',
    config_dict={
        'source': 'copernicus',
        'currents': 'total',
        'USERNAME': 'mmariuswiggert',
        'PASSWORD': 'tamku3-qetroR-guwneq',
        'DATASET_ID': 'cmems_mod_glo_phy_anfc_merged-uv_PT1H-i'
    }
)
print(f'Init Source: {time.time()-start}s')

##### Initialize Platform #####
start = time.time()
init_state = PlatformState(
    date_time=t_0,
    lon=units.Distance(deg=-81.5),
    lat=units.Distance(deg=23.5)
)
platform_specs = PlatformSpecs(
    battery_cap=units.Energy(watt_hours=1000),
    u_max=units.Velocity(mps=0.1),
    motor_efficiency=1.0
)
platform = Platform(state=init_state, source=source, specs=platform_specs)
print(f'Init Platform: {time.time()-start}s')

trajectory = np.zeros((steps, 2))
##### Run Simulation #####
for step in tqdm(range(steps)):
    state = platform.simulate_step(action=[1,3.14/2], time_delta=stride)
    trajectory[step] = np.array([state.lon.deg, state.lat.deg])

##### Plotting using old Utils #####
start = time.time()
data_store = utils.simulation_utils.copernicusmarine_datastore('cmems_mod_glo_phy_anfc_merged-uv_PT1H-i','mmariuswiggert','tamku3-qetroR-guwneq')
DS_currents = xr.open_dataset(data_store)[['uo', 'vo']].isel(depth=0)
file_dicts = {'data_source_type': 'cop_opendap', 'content': DS_currents, 'grid_dict': Problem.derive_grid_dict_from_xarray(DS_currents)}
plotting_utils.plot_2D_traj_over_currents(
    x_traj=trajectory.T,
    deg_around_x0_xT_box=2,
    time=t_0.timestamp(),
    file_dicts=file_dicts
)
print(f'Plotting: {time.time()-start}s')

print(f'Total Script: {time.time()-script_start}s')