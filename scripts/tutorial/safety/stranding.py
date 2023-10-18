#%%

# TODO: Design mission that will strand
# -> Make it run fast: long time between updates to compute less
# Area with plenty of land
import datetime

from tqdm import tqdm

from ocean_navigation_simulator.controllers.NaiveController import (
    NaiveController,
)
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)

# from ocean_navigation_simulator.environment.Arena import Arena
# TODO: would be great to have heatmap of strandings
# TODO: heatmap of places from where we start to strand (e..g n starts, 1 target, see which starts lead to stranding)
from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.utils import units

# # %% Test 1a: start on land
# # No bathymetry
arena = ArenaFactory.create(scenario_name="gulf_of_mexico_HYCOM_hindcast_local")

x_0 = PlatformState(
    lon=units.Distance(deg=-80.7),
    lat=units.Distance(deg=25.4),
    date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
)
x_T = SpatialPoint(lon=units.Distance(deg=-80.3), lat=units.Distance(deg=24.6))

problem = NavigationProblem(
    start_state=x_0,
    end_region=x_T,
    target_radius=0.1,
)

planner = NaiveController(problem)
observation = arena.reset(platform_state=x_0)

for i in tqdm(range(int(3600 * 24 * 5 / 600))):  # 3 days
    action = planner.get_action(observation=observation)
    observation = arena.step(action)
    problem_status = arena.problem_status(problem=problem)
    if problem_status != 0:
        print(f"Problem status: {problem_status}")
        break

assert problem_status == -2

# %% Test 1b: start on land
# With bathymetry
# arena = ArenaFactory.create(scenario_name="safety_gulf_of_mexico_HYCOM_hindcast_local")

# x_0 = PlatformState(
#     lon=units.Distance(deg=-80.7),
#     lat=units.Distance(deg=25.4),
#     date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
# )
# x_T = SpatialPoint(lon=units.Distance(deg=-80.3), lat=units.Distance(deg=24.6))

# problem = NavigationProblem(
#     start_state=x_0,
#     end_region=x_T,
#     target_radius=0.1,
# )

# planner = NaiveController(problem)
# observation = arena.reset(platform_state=x_0)

# for i in tqdm(range(int(3600 * 24 * 5 / 600))):  # 3 days
#     action = planner.get_action(observation=observation)
#     observation = arena.step(action)
#     problem_status = arena.problem_status(problem=problem)
#     if problem_status != 0:
#         print(f"Problem status: {problem_status}")
#         break

# assert problem_status == -2
#%% Plot the arena trajectory on the map
# arena.plot_all_on_map(problem=problem)

# %% Test 2: End on land
# arena = ArenaFactory.create(scenario_name="safety_gulf_of_mexico_HYCOM_hindcast_local")
# x_0 = PlatformState(
#     lon=units.Distance(deg=-81.35),
#     lat=units.Distance(deg=25.5),
#     date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
# )
# x_T = SpatialPoint(lon=units.Distance(deg=-80.7), lat=units.Distance(deg=25.4))

# problem = NavigationProblem(
#     start_state=x_0,
#     end_region=x_T,
#     target_radius=0.1,
# )

# planner = NaiveController(problem)
# observation = arena.reset(platform_state=x_0)

# for i in tqdm(range(int(3600 * 24 * 5 / 600))):  # 3 days
#     action = planner.get_action(observation=observation)
#     observation = arena.step(action)
#     problem_status = arena.problem_status(problem=problem)
#     if problem_status != 0:
#         print(f"Problem status: {problem_status}")
#         break

# assert problem_status == -2


#%% Plot the arena trajectory on the map
arena.plot_all_on_map(problem=problem, show_control_trajectory=False)
