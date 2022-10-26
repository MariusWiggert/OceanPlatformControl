"""
    Test Script to test integration of simplex noise.

    Currently:
        - create Problem with available Copernicus FC / HHYCOM HC
        - create Arena and download required files for problem
        - runs arena until termination

    ToDos:
        - change config.yaml to specify:
            - use of Simplex Noise
            - seed
        - create SimplexOceanFiled(OceanField):
            - this class should have the same interface as OceanField
                - takes configuration dictionaries
                - has properties hindcast_data_source, forecast_data_source
        - create SimplexForecastSource(OceanField):

        - add additional requirements to requirements.txt for simple installation (so we can add them to the main file later)
"""
import logging

from ocean_navigation_simulator.controllers.NaiveController import (
    NaiveController,
)
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.reinforcement_learning.missions.FileProblemFactory import (
    FileProblemFactory,
)
from ocean_navigation_simulator.utils.misc import set_arena_loggers, timing

with timing("Script running for {:.1f}s"):
    set_arena_loggers(logging.INFO)

    # Step 1: Create Problem, Controller & Arena
    problem_factory = FileProblemFactory(
        csv_file="scripts/test_simplex_noise/valid_problem_from_generation/problems.csv"
    )
    problem = problem_factory.next_problem()
    controller = NaiveController(problem=problem)
    # Download Files to /tmp/*
    arena = ArenaFactory.create(
        scenario_file="scripts/test_simplex_noise/gulf_of_mexico_Copernicus_forecast_Simplex_hindcast.yaml",
        scenario_config={
            "ocean_dict": {
                "forecast": {
                    "source_settings": {
                        "seed": 2022,
                    }
                }
            }
        },
        problem=problem,
    )
    arena.collect_trajectory = False

    # Step 2: Run Simulation until termination
    observation = arena.reset(problem.start_state)

    with timing("Arena running for {:.1f}s"):
        while problem.is_done(observation.platform_state) == 0:
            observation = arena.step(controller.get_action(observation))

    print(
        "Simulation finished: {status}, {passed:.1f}h".format(
            status=arena.problem_status_text(arena.problem_status(problem=problem)),
            passed=problem.passed_seconds(arena.platform.state) / 3600,
        )
    )
    arena.plot_all_on_map(problem=problem)
