import logging

# import numpy as np
# import seaborn as sns
# from matplotlib import pyplot as plt
# from mpl_toolkits import mplot3d

from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.reinforcement_learning.missions.FileProblemFactory import (
    FileProblemFactory,
)

# from ocean_navigation_simulator.reinforcement_learning.RLController import (
#     RLController,
# )
from ocean_navigation_simulator.utils.misc import set_arena_loggers, timing

with timing("Script running for {}"):
    generation = "/seaweed-storage/generation/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/divers_training_improved_2022_10_23_05_10_12/"
    experiment = "/seaweed-storage/experiments/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/divers_training_2022_10_24_13_35_20/"

    set_arena_loggers(logging.INFO)

    problem_factory = FileProblemFactory(generation + "problems.csv", filter={"indices": [145701]})
    problems = problem_factory.get_problem_list(8)
    problem = problems[0]

    print(problem)

    # rl_controller = RLController(
    #     config={"experiment": experiment, 'checkpoint': 84},
    #     problem=problem,
    #     arena=arena,
    # )

    # point=problem.start_state
    # width = 100
    # width_deg=2
    # int = cached_planner.interpolate_value_function_in_hours(
    #     point=point,
    #     width=width,
    #     width_deg=width_deg,
    #     allow_spacial_extrapolation=True,
    #     allow_temporal_extrapolation=True,
    # )
    # out_x = np.linspace(point.lon.deg - width_deg / 2, point.lon.deg + width_deg / 2, width)
    # out_y = np.linspace(point.lat.deg - width_deg / 2, point.lat.deg + width_deg / 2, width)
    #
    # fig, ax = plt.subplots()
    # [X, Y] = np.meshgrid(out_x, out_y)
    # c = ax.contourf(X, Y, int, levels=50)
    # for problem in [problem]:
    #     ax.scatter(
    #         problem.start_state.lon.deg,
    #         problem.start_state.lat.deg,
    #         facecolors="black" if problem.extra_info["random"] else "none",
    #         edgecolors="black" if problem.extra_info["random"] else "r",
    #         marker="o",
    #         label="starts",
    #     )
    # ax.scatter(
    #     problem.end_region.lon.deg,
    #     problem.end_region.lat.deg,
    #     facecolors="red",
    #     marker="x",
    #     label="target",
    # )
    # fig.colorbar(c, ax=ax)
    # plt.show()
    # cached_planner.interpolate_value_function_in_hours(
    #     point=SpatioTemporalPoint(
    #         lon=problem.end_region.lon,
    #         lat=problem.end_region.lat,
    #         date_time=problem.start_state.date_time,
    #     )
    # )

    arena = ArenaFactory.create(
        scenario_file="config/reinforcement_learning/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast.yaml",
        problem=problem,
    )
    hindcast_planner = problem.get_cached_hindcast_planner(generation, arena)
    obs = arena.reset(problem.start_state)
    while arena.problem_status(problem) == 0:
        obs = arena.step(
            hindcast_planner.get_action(
                obs.replace_datasource(arena.ocean_field.hindcast_data_source)
            )
        )
    ax = arena.plot_all_on_map(problem=problem, margin=0.25, return_ax=True)

    arena = ArenaFactory.create(
        scenario_file="config/reinforcement_learning/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast.yaml",
        problem=problem,
    )
    forecast_planner = problem.get_cached_forecast_planner(generation, arena)
    obs = arena.reset(problem.start_state)
    while arena.problem_status(problem) == 0:
        obs = arena.step(forecast_planner.get_action(obs))
    arena.plot_all_on_map(background=None, ax=ax)
