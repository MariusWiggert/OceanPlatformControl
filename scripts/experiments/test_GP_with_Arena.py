import datetime
import numpy as np
import time
import math

from ocean_navigation_simulator.env.Arena import Arena, ArenaObservation
from ocean_navigation_simulator.env.Observer import Observer
from ocean_navigation_simulator.env.Platform import Platform, PlatformState
from ocean_navigation_simulator.env.PlatformState import SpatialPoint
from ocean_navigation_simulator.env.controllers.unmotorized_controller import UnmotorizedController
from ocean_navigation_simulator.env.problem import Problem

from ocean_navigation_simulator.env.utils import units
from scripts.experiments.class_gp import OceanCurrentGP

_DELTA_TIME_NEW_PREDICTION = datetime.timedelta(hours=1)
_DURATION_SIMULATION = datetime.timedelta(days=3)
_NUMBER_STEPS = int(math.ceil(_DURATION_SIMULATION.total_seconds()/_DELTA_TIME_NEW_PREDICTION.total_seconds()))
_N_DATA_PTS = 100 #TODO: FINE TUNE THAT




#for each hour:
    # get the forecasts
    # Give it to the observer
    # The observer compute using the GP




#%%
initial_position = (-90.963268, 26.519735)
platform_state = PlatformState(
    #date_time=datetime.datetime(year=2021, month=11, day=20, hour=12, minute=0, second=0, tzinfo=datetime.timezone.utc),
    date_time=datetime.datetime(2021, 11, 26, 23, 30, tzinfo=datetime.timezone.utc),
    lon=units.Distance(deg=initial_position[0]),
    lat=units.Distance(deg=initial_position[1]),
    seaweed_mass=units.Mass(kg=0),
    battery_charge=units.Energy(watt_hours=100)
)
#%%
sim_cache_dict = {
    'deg_around_x_t': 1,
    'time_around_x_t': 3600 * 24 * 3
}
platform_dict = {
    'battery_cap_in_wh': 400.0,
    'u_max_in_mps': 0.1,
    'motor_efficiency': 1,
    'solar_panel_size': 0.5,
    'solar_efficiency': 0.2,
    'drag_factor': 0, # We assume for now that the platform only follows the currents
    'dt_in_s': _DELTA_TIME_NEW_PREDICTION.total_seconds(), # we want to refresh every hour
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
    }
}

solar_source_dict = {
    'field': 'SolarIrradiance',
    'source': 'analytical_w_caching',
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

forecast_file_config_dict = {'source': 'forecast_files',
                'subset_time_buffer_in_s': 4000,
                'source_settings': {
                    'folder': "data/forecast_test/"
                }
                             }

arena = Arena(
    sim_cache_dict=sim_cache_dict,
    platform_dict=platform_dict,
    ocean_dict={'hindcast': ocean_source_dict, 'forecast': forecast_file_config_dict},
    solar_dict={'hindcast': solar_source_dict, 'forecast': None}
)

#%%
controller = UnmotorizedController(problem=Problem(
    start_state=platform_state,
    end_region=PlatformState(
        date_time=platform_state.date_time,
        lon=units.Distance(deg=initial_position[0]),
        lat=units.Distance(deg=initial_position[1])
    )
))
#%%
observation = arena.reset(platform_state)
#%%
gp = OceanCurrentGP(arena.ocean_field)
observer = Observer(gp, arena)

#%%
start = time.time()
for i in range(_NUMBER_STEPS):
    print("step",i)
    action = controller.get_action(observation)
    arena_observation = arena.step(action)
    observer.observe(observation.platform_state, [0.5, 0.3])
    # Give as input to observer: arena.ocean_field.get_forecast() - Ground_truth(), position
    #get_forecasts(observation.platform_state)
print("total: ", time.time() - start)
# Testing if solar caching or not-caching makes much of a difference
# For 240 steps: without caching 0.056s > with caching: 0.037.
#%%
arena.do_nice_plot(x_T=np.array([controller.problem.end_region.lon.deg, controller.problem.end_region.lat.deg]))

#%%
print("over")