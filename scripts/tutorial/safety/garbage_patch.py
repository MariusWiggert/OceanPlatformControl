#%%
import datetime
from ocean_navigation_simulator.controllers.NaiveController import NaiveController
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory

from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from tqdm import tqdm

# Test using gulf of mexico with fake garbage data
# # %% Test 1a: start on garbage
# arena = ArenaFactory.create(scenario_name="safety_gulf_of_mexico_HYCOM_hindcast_local")

# x_0 = PlatformState(
#     lon=units.Distance(deg=-86),
#     lat=units.Distance(deg=26),
#     date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
# )
# x_T = SpatialPoint(lon=units.Distance(deg=-88), lat=units.Distance(deg=27))

# problem = NavigationProblem(
#     start_state=x_0,
#     end_region=x_T,
#     target_radius=0.1,
# )

# planner = NaiveController(problem)
# observation = arena.reset(platform_state=x_0)

# for i in tqdm(range(int(3600 * 24 * 5 / 600))):  # 5 days
#     action = planner.get_action(observation=observation)
#     observation = arena.step(action)
#     problem_status = arena.problem_status(problem=problem)
#     if problem_status != 0:
#         print(f"Problem status: {problem_status}")
#         break

# assert problem_status == -4


# %% Test 1b: no garbage
# arena = ArenaFactory.create(scenario_name="safety_gulf_of_mexico_HYCOM_hindcast_local")

# x_0 = PlatformState(
#     lon=units.Distance(deg=-86),
#     lat=units.Distance(deg=22),
#     date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
# )
# x_T = SpatialPoint(lon=units.Distance(deg=-88), lat=units.Distance(deg=27))

# problem = NavigationProblem(
#     start_state=x_0,
#     end_region=x_T,
#     target_radius=0.1,
# )

# planner = NaiveController(problem)
# observation = arena.reset(platform_state=x_0)

# for i in tqdm(range(int(3600 * 24 * 0.1 / 600))):  # 0.1 days
#     action = planner.get_action(observation=observation)
#     observation = arena.step(action)
#     problem_status = arena.problem_status(problem=problem)
#     if problem_status != 0:
#         print(f"Problem status: {problem_status}")
#         break

# assert problem_status == 0

# # %% Test 2: Go through garbage (end there)
# arena = ArenaFactory.create(scenario_name="safety_gulf_of_mexico_HYCOM_hindcast_local")

# x_0 = PlatformState(
#     lon=units.Distance(deg=-86),
#     lat=units.Distance(deg=24),
#     date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
# )
# x_T = SpatialPoint(lon=units.Distance(deg=-86), lat=units.Distance(deg=26))

# problem = NavigationProblem(
#     start_state=x_0,
#     end_region=x_T,
#     target_radius=0.1,
# )

# planner = NaiveController(problem)
# observation = arena.reset(platform_state=x_0)

# for i in tqdm(range(int(3600 * 24 * 5 / 600))):  # 5 days
#     action = planner.get_action(observation=observation)
#     observation = arena.step(action)
#     problem_status = arena.problem_status(problem=problem)
#     if problem_status != 0:
#         # Stop when encountering garbage
#         print(f"Problem status: {problem_status}")
#         break

# assert problem_status == -4

# %% Test 3: Start outside garbage, go in, check trajectory
arena = ArenaFactory.create(scenario_name="safety_gulf_of_mexico_HYCOM_hindcast_local")

x_0 = PlatformState(
    lon=units.Distance(deg=-87),
    lat=units.Distance(deg=24.9),
    date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
)
x_T = SpatialPoint(lon=units.Distance(deg=-87), lat=units.Distance(deg=27))

problem = NavigationProblem(
    start_state=x_0,
    end_region=x_T,
    target_radius=0.1,
)

planner = NaiveController(problem)
observation = arena.reset(platform_state=x_0)

amount_of_garbage = 0
for i in tqdm(range(int(3600 * 24 * 5 / 600))):  # 5 days
    action = planner.get_action(observation=observation)
    observation = arena.step(action)
    problem_status = arena.problem_status(problem=problem)
    if problem_status != 0:
        # Continue when encountering garbage, else break
        if problem_status == -4:
            amount_of_garbage += 1
        print(f"Problem status: {problem_status}")
        if amount_of_garbage > 20:
            break

print(arena.state_trajectory)
print(arena.state_trajectory[-1][-1])
print(f"{arena.state_trajectory[-1][-1]:.64f}")
assert arena.state_trajectory[-1][-1] != 0.0


# TODO: plot with garbage and traj
