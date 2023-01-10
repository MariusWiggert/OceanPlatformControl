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
            "region": "GOM",
            "hindcast": {
                "field": "OceanCurrents",
                "source": "hindcast_files",
                "source_settings": {
                    "local": True,
                    "folder": "data/tests/test_BathymetrySource/",
                    "source": "HYCOM",
                    "type": "hindcast",
                },
            },
            "forecast": None,
        },
        "bathymetry_dict": {
            "field": "Bathymetry",
            "source": "gebco",
            "source_settings": {
                "filepath": "ocean_navigation_simulator/package_data/bathymetry_and_garbage/bathymetry_global_res_0.083_0.083_max.nc"
            },
            "casadi_cache_settings": {"deg_around_x_t": 20},
            "use_geographic_coordinate_system": True,
        },
        "garbage_dict": None,
        "solar_dict": {"hindcast": None, "forecast": None},
        "seaweed_dict": {"hindcast": None, "forecast": None},
    }
    t_interval = [datetime.datetime(2021, 11, 24, 0, 0, 0), datetime.datetime(2021, 12, 5, 0, 0, 0)]
    # download files if not there yet
    ArenaFactory.download_required_files(
        archive_source="HYCOM",
        archive_type="hindcast",  # should be hindcast once that works on C3
        region="GOM",
        download_folder="data/tests/test_BathymetrySource/",
        t_interval=t_interval,
    )
    return scenario_config


@pytest.mark.parametrize(
    "start, stop, expected_problem_status",
    [([-80.7, 25.4], [-80.3, 24.6], -2)],
)
def test__is_on_land__start_on_land_with_bathymetry(
    scenario_config, start, stop, expected_problem_status
):
    arena = ArenaFactory.create(scenario_config=scenario_config)
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
    "start, stop, expected_problem_status",
    [([-81.35, 25.5], [-80.7, 25.4], -2)],
)
def test__is_on_land__go_onto_land_with_bathymetry(
    scenario_config, start, stop, expected_problem_status
):
    arena = ArenaFactory.create(scenario_config=scenario_config)
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
    "start, stop, expected_problem_status, expected_bathymetry_source",
    [([-80.7, 25.4], [-80.3, 24.6], -2, None)],
)
def test__is_on_land__start_on_land_no_bathymetry(
    scenario_config, start, stop, expected_problem_status, expected_bathymetry_source
):
    # To be able to reuse the fixture we do this in a hacky way.
    # This test HAS TO BE the last one, as it will otherwise override the fixture
    scenario_config["bathymetry_dict"] = None
    arena = ArenaFactory.create(scenario_config=scenario_config)
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
