import logging

from ocean_navigation_simulator.controllers.NaiveController import NaiveController
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.reinforcement_learning.missions.FileProblemFactory import FileProblemFactory
from ocean_navigation_simulator.utils.misc import timing, set_arena_loggers

import seaborn as sns
sns.set_theme()

with timing("Script running for {}"):
    generation = "/seaweed-storage/generation/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/divers_training_improved_2022_10_23_05_10_12/"
    experiment = "/seaweed-storage/experiments/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/divers_training_2022_10_24_13_35_20/"

    set_arena_loggers(logging.INFO)

    problem_factory = FileProblemFactory(
        generation + "problems.csv",
        filter={
            "no_random": True
        }
    )
    problem = problem_factory.next_problem()
    print(problem)

    arena = ArenaFactory.create(
        scenario_file="config/reinforcement_learning/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast.yaml",
        problem=problem,
    )
    controller = NaiveController(problem=problem)
    obs = arena.reset(problem.start_state)
    while arena.problem_status(problem) == 0:
        obs = arena.step(controller.get_action(obs))

    ax = arena.plot_all_on_map(
        problem=problem,
        margin=0.25,
        show_current_position=False,
        control_stride=100,
        return_ax=True,
    )
    fig = ax.get_figure()
    fig.savefig('plots/3_introduction/problem_statement.png', dpi=300)
    fig.show()