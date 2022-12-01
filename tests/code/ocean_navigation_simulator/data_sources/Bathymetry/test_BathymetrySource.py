import datetime

import pytest

from ocean_navigation_simulator.controllers.NaiveController import (
    NaiveController,
)
from ocean_navigation_simulator.data_sources.Bathymetry.BathymetrySource import (
    BathymetrySource2d,
)
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.utils import units

# TODO: In theory I can run most through parametrization, but then I do not know what failure I actually test...
# TODO: actually we are testing arena...


@pytest.mark.parametrize(
    "scenario_name, start, stop, expected_problem_status, expected_bathymetry_source",
    [("gulf_of_mexico_HYCOM_hindcast_local", [-80.7, 25.4], [-80.3, 24.6], -2, None)],
)
def test__is_on_land__start_on_land_no_bathymetry(
    scenario_name, start, stop, expected_problem_status, expected_bathymetry_source
):
    arena = ArenaFactory.create(scenario_name=scenario_name)
    assert arena.bathymetry_source == expected_bathymetry_source
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
            assert problem_status == expected_problem_status
            break


@pytest.mark.parametrize(
    "scenario_name, start, stop, expected_problem_status",
    [("safety_gulf_of_mexico_HYCOM_hindcast_local", [-80.7, 25.4], [-80.3, 24.6], -2)],
)
def test__is_on_land__start_on_land_with_bathymetry(
    scenario_name, start, stop, expected_problem_status
):
    arena = ArenaFactory.create(scenario_name=scenario_name)
    assert isinstance(arena.bathymetry_source, BathymetrySource2d)
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
            assert problem_status == expected_problem_status
            break


@pytest.mark.parametrize(
    "scenario_name, start, stop, expected_problem_status",
    [("safety_gulf_of_mexico_HYCOM_hindcast_local", [-81.35, 25.5], [-80.7, 25.4], -2)],
)
def test__is_on_land__go_onto_land_with_bathymetry(
    scenario_name, start, stop, expected_problem_status
):
    arena = ArenaFactory.create(scenario_name=scenario_name)
    assert isinstance(arena.bathymetry_source, BathymetrySource2d)
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
            assert problem_status == expected_problem_status
            break
