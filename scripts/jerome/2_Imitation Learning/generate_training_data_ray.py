import pandas as pd
from datetime import datetime
import gc
import ray
import time

from ocean_navigation_simulator.env.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.env.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.env.Platform import PlatformState
from ocean_navigation_simulator.env.PlatformState import SpatialPoint
from ocean_navigation_simulator.env.utils import units
from ocean_navigation_simulator.reinforcement_learning.scripts import generate_training_data_for_imitation

script_start_time = time.time()

print('Script started ...')

ray.init("ray://13.68.187..126:10001")

print('Ray initialised ...')

@ray.remote(num_cpus=1)
def generate_data_for_mission(index):
    mission_df = pd.read_csv('./data/value_function_learning/missions.csv', index_col=0)
    row = mission_df.iloc[index]

    print('##############################')
    print(f'## Mission {index:03d}             ###')
    print(f'## ({time.time() - script_start_time:.1f}s)')
    print('##############################')
    start = time.time()
    arena = ArenaFactory.create(scenario_name='gulf_of_mexico_HYCOM_forecast_Copernicus_hindcast')
    problem = NavigationProblem(
        start_state=PlatformState(
            lon=units.Distance(deg=row['x_0_lon']),
            lat=units.Distance(deg=row['x_0_lat']),
            date_time=datetime.fromisoformat(row['t_0'])
        ),
        end_region=SpatialPoint(
            lon=units.Distance(deg=row['x_T_lon']),
            lat=units.Distance(deg=row['x_T_lat'])
        ),
        target_radius=0.1,
        timeout=100*3600
    )
    print(f'Arena & Problem created ({time.time() - start:.1f}s)')
    generate_training_data_for_imitation(
        arena=arena,
        problem=problem,
        mission_folder=f'data/value_function_learning/mission_{index}',
        steps=600,
        verbose=1,
        plot_after=1000,
    )
    # Delete Objects
    del arena
    del problem
    gc.collect()

    return index


mission_df = pd.read_csv('./data/value_function_learning/missions.csv', index_col=0)
number_of_missions = 1 #len(mission_df.index)

futures = [generate_data_for_mission.remote(i) for i in range(108, 109)]
print(futures)

print('Starting ray ...')

print(ray.get(futures))

print(f"Total Script Time: {time.time()-script_start_time:.2f}s = {(time.time()-script_start_time)/60:.2f}min")