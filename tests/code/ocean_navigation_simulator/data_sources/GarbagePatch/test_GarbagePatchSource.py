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


# Might crash due to wrong meta data, other than that it works
@pytest.fixture(scope="module")
def scenario_config():
    scenario_config = {
        "casadi_cache_dict": {"deg_around_x_t": 0.5, "time_around_x_t": 36000.0},
        "platform_dict": {
            "battery_cap_in_wh": 400.0,
            "u_max_in_mps": 0.1,
            "motor_efficiency": 1.0,
            "solar_panel_size": 0.5,
            "solar_efficiency": 0.2,
            "drag_factor": 675.0,
            "dt_in_s": 600.0,
        },
        "use_geographic_coordinate_system": True,
        "spatial_boundary": None,
        "timeout": 259200,
        "ocean_dict": {
            "region": "Region 1",
            "hindcast": {
                "field": "OceanCurrents",
                "source": "hindcast_files",
                "source_settings": {
                    "local": True,
                    "folder": "data/tests/test_GarbagePatchSource/",
                    "source": "HYCOM",
                    "type": "hindcast",
                },
            },
            "forecast": None,
        },
        "bathymetry_dict": None,
        "garbage_dict": {
            "field": "Garbage",
            "source": "Lebreton",
            "source_settings": {
                "filepath": "ocean_navigation_simulator/package_data/bathymetry_and_garbage/garbage_patch_global_res_0.083_0.083.nc"
            },
            "casadi_cache_settings": {"deg_around_x_t": 10},
            "use_geographic_coordinate_system": True,
        },
        "solar_dict": {"hindcast": None, "forecast": None},
        "seaweed_dict": {"hindcast": None, "forecast": None},
    }
    t_interval = [datetime.datetime(2022, 10, 4, 0, 0, 0), datetime.datetime(2022, 10, 7, 0, 0, 0)]
    # download files if not there yet
    ArenaFactory.download_required_files(
        archive_source="HYCOM",
        archive_type="hindcast",
        region="Region 1",
        download_folder=scenario_config["ocean_dict"]["hindcast"]["source_settings"]["folder"],
        t_interval=t_interval,
    )
    return scenario_config


@pytest.mark.parametrize(
    "start, stop, expected_problem_status",
    [([-158, 29], [-158, 30], -4)],
)
def test__is_on_garbage__start_on_garbage_with_garbage_source(
    scenario_config, start, stop, expected_problem_status
):
    arena = ArenaFactory.create(scenario_config=scenario_config)
    assert isinstance(arena.garbage_source, GarbagePatchSource2d)
    x_0 = PlatformState(
        lon=units.Distance(deg=start[0]),
        lat=units.Distance(deg=start[1]),
        date_time=datetime.datetime(2022, 10, 4, 0, 0, 0, tzinfo=datetime.timezone.utc),
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
    "start, stop, expected_problem_status",
    [([-158, 28], [-158, 30], 0)],
)
def test__is_on_garbage__no_garbage_with_garbage_source(
    scenario_config, start, stop, expected_problem_status
):
    arena = ArenaFactory.create(scenario_config=scenario_config)
    assert isinstance(arena.garbage_source, GarbagePatchSource2d)
    x_0 = PlatformState(
        lon=units.Distance(deg=start[0]),
        lat=units.Distance(deg=start[1]),
        date_time=datetime.datetime(2022, 10, 4, 0, 0, 0, tzinfo=datetime.timezone.utc),
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
    "start, stop, expected_problem_status",
    [([-159.15, 28.5], [-158, 29], -4)],
)
def test__is_on_garbage__go_into_garbage_with_garbage_source(
    scenario_config, start, stop, expected_problem_status
):
    # Also test trajectory
    arena = ArenaFactory.create(scenario_config=scenario_config)
    assert isinstance(arena.garbage_source, GarbagePatchSource2d)
    x_0 = PlatformState(
        lon=units.Distance(deg=start[0]),
        lat=units.Distance(deg=start[1]),
        date_time=datetime.datetime(2022, 10, 4, 0, 0, 0, tzinfo=datetime.timezone.utc),
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


@pytest.mark.parametrize(
    "start, stop, expected_problem_status",
    [([-158, 29], [-158, 30], 0)],
)
def test__is_on_garbage__start_on_garbage_no_garbage_source(
    scenario_config, start, stop, expected_problem_status
):
    # Hacky overwrite so we can use the fixture.
    # This test has to be last to not overrride the garbage_dict.
    scenario_config["garbage_dict"] = None
    arena = ArenaFactory.create(scenario_config=scenario_config)
    assert arena.garbage_source is None
    x_0 = PlatformState(
        lon=units.Distance(deg=start[0]),
        lat=units.Distance(deg=start[1]),
        date_time=datetime.datetime(2022, 10, 4, 0, 0, 0, tzinfo=datetime.timezone.utc),
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
