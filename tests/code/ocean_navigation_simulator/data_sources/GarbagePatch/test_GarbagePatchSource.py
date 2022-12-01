import datetime

import pytest

from ocean_navigation_simulator.controllers.NaiveController import (
    NaiveController,
)
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator.data_sources.GarbagePatch.GarbagePatchSource import (
    GarbagePatchSource2d,
)


# Test using gulf of mexico with fake garbage data
@pytest.mark.parametrize(
    "scenario_name, start, stop, expected_problem_status",
    [("safety_gulf_of_mexico_HYCOM_hindcast_local", [-86, 26], [-88, 27], -4)],
)
def test__is_on_garbage__start_on_garbage_with_garbage_source(
    scenario_name, start, stop, expected_problem_status
):
    arena = ArenaFactory.create(scenario_name=scenario_name)
    assert isinstance(arena.garbage_source, GarbagePatchSource2d)
    x_0 = PlatformState(
        lon=units.Distance(deg=start[0]),
        lat=units.Distance(deg=start[1]),
        date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
    )
    x_T = SpatialPoint(lon=units.Distance(deg=stop[0]), lat=units.Distance(deg=stop[1]))

    problem = NavigationProblem(
        start_state=x_0,
        end_region=x_T,
        target_radius=0.1,
    )

    planner = NaiveController(problem)
    observation = arena.reset(platform_state=x_0)

    for i in range(int(3600 * 24 * 5 / 600)):  # 5 days
        action = planner.get_action(observation=observation)
        observation = arena.step(action)
        problem_status = arena.problem_status(problem=problem)
        if problem_status != 0:
            break

    assert problem_status == expected_problem_status


@pytest.mark.parametrize(
    "scenario_name, start, stop, expected_problem_status",
    [("gulf_of_mexico_HYCOM_hindcast_local", [-86, 26], [-88, 27], 0)],
)
def test__is_on_garbage__start_on_garbage_no_garbage_source(
    scenario_name, start, stop, expected_problem_status
):
    arena = ArenaFactory.create(scenario_name=scenario_name)
    assert arena.garbage_source is None
    x_0 = PlatformState(
        lon=units.Distance(deg=start[0]),
        lat=units.Distance(deg=start[1]),
        date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
    )
    x_T = SpatialPoint(lon=units.Distance(deg=stop[0]), lat=units.Distance(deg=stop[1]))

    problem = NavigationProblem(
        start_state=x_0,
        end_region=x_T,
        target_radius=0.1,
    )

    planner = NaiveController(problem)
    observation = arena.reset(platform_state=x_0)

    for i in range(int(3600 * 24 * 0.25 / 600)):  # 0.25 days
        action = planner.get_action(observation=observation)
        observation = arena.step(action)
        problem_status = arena.problem_status(problem=problem)
        if problem_status != 0:
            break

    assert problem_status == expected_problem_status


@pytest.mark.parametrize(
    "scenario_name, start, stop, expected_problem_status",
    [("safety_gulf_of_mexico_HYCOM_hindcast_local", [-86, 22], [-88, 27], 0)],
)
def test__is_on_garbage__no_garbage_with_garbage_source(
    scenario_name, start, stop, expected_problem_status
):
    arena = ArenaFactory.create(scenario_name=scenario_name)
    assert isinstance(arena.garbage_source, GarbagePatchSource2d)
    x_0 = PlatformState(
        lon=units.Distance(deg=start[0]),
        lat=units.Distance(deg=start[1]),
        date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
    )
    x_T = SpatialPoint(lon=units.Distance(deg=stop[0]), lat=units.Distance(deg=stop[1]))

    problem = NavigationProblem(
        start_state=x_0,
        end_region=x_T,
        target_radius=0.1,
    )

    planner = NaiveController(problem)
    observation = arena.reset(platform_state=x_0)

    for i in range(int(3600 * 24 * 0.25 / 600)):  # 0.25 days
        action = planner.get_action(observation=observation)
        observation = arena.step(action)
        problem_status = arena.problem_status(problem=problem)
        if problem_status != 0:
            break

    assert problem_status == expected_problem_status


@pytest.mark.parametrize(
    "scenario_name, start, stop, expected_problem_status",
    [("safety_gulf_of_mexico_HYCOM_hindcast_local", [-87, 24.9], [-87, 27], -4)],
)
def test__is_on_garbage__go_into_garbage_with_garbage_source(
    scenario_name, start, stop, expected_problem_status
):
    # Also test trajectory
    arena = ArenaFactory.create(scenario_name=scenario_name)
    assert isinstance(arena.garbage_source, GarbagePatchSource2d)
    x_0 = PlatformState(
        lon=units.Distance(deg=start[0]),
        lat=units.Distance(deg=start[1]),
        date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
    )
    x_T = SpatialPoint(lon=units.Distance(deg=stop[0]), lat=units.Distance(deg=stop[1]))

    problem = NavigationProblem(
        start_state=x_0,
        end_region=x_T,
        target_radius=0.1,
    )

    planner = NaiveController(problem)
    observation = arena.reset(platform_state=x_0)

    # Garbage in state_trajectory is one time step delayed to problem_status
    # -> This is why we test with steps in garbage
    steps_in_garbage = 0
    for i in range(int(3600 * 24 * 5 / 600)):  # 5 days
        action = planner.get_action(observation=observation)
        observation = arena.step(action)
        problem_status = arena.problem_status(problem=problem)
        if problem_status != 0:
            if problem_status == -4:
                steps_in_garbage += 1
            if steps_in_garbage > 1:
                break

    assert problem_status == expected_problem_status
    # Need to have a higher value than 0. 0 (bool) means no garbage, higher value means garbage
    assert arena.state_trajectory[-1][5] > 0
