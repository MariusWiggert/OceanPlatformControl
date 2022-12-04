import logging

import numpy as np
import ray
from matplotlib import pyplot as plt

from ocean_navigation_simulator.controllers.NaiveController import NaiveController
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.reinforcement_learning.missions.FileProblemFactory import FileProblemFactory
from ocean_navigation_simulator.utils.misc import timing, set_arena_loggers
from ocean_navigation_simulator.utils import cluster_utils, units

import seaborn as sns
sns.set_theme()

cluster_utils.init_ray()

with timing("Script running for {}"):
    generation = "/seaweed-storage/generation/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/divers_training_improved_2022_10_23_05_10_12/"
    # generation = "./"

    set_arena_loggers(logging.INFO)

    problem_factory = FileProblemFactory(
        generation + "problems.csv",
        filter={
            "no_random": True,
            "starts_per_target": 1,
        }
    )

    dt_in_s = [10, 60, 600, 3600]
    labels = ['1min', '10min', '1h']

    @ray.remote(resources={'RAM': 4000})
    def distance(problem):
        results = []

        for i, dt in enumerate(dt_in_s):
            arena = ArenaFactory.create(
                scenario_file="config/reinforcement_learning/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast.yaml",
                scenario_config={
                    'platform_dict': {
                        'dt_in_s': dt
                    },
                    # 'ocean_dict': {
                    #     'hindcast': {
                    #         'source_settings': {
                    #             'local': False
                    #         }
                    #
                    #     },
                    #     'forecast': {
                    #         'source_settings': {
                    #             'local': False
                    #         }
                    #
                    #     }
                    # }
                },
                problem=problem,
            )
            controller = NaiveController(problem=problem)
            obs = arena.reset(problem.start_state)
            while arena.problem_status(problem) == 0:
                obs = arena.step(controller.get_action(obs))

            if i == 0:
                best_arena = arena
            else:
                # compare = np.repeat(best_arena.state_trajectory[None, :, :2], arena.state_trajectory.shape[0], axis=0)
                # own = np.repeat(arena.state_trajectory[:, None, :2], best_arena.state_trajectory.shape[0], axis=1)
                # distances = np.linalg.norm(own - compare, axis=2).min(axis=1)
                distances = np.zeros((arena.state_trajectory.shape[0]))
                for i in range(arena.state_trajectory.shape[0]):
                    distances[i] = np.linalg.norm(best_arena.state_trajectory[:, :2] - arena.state_trajectory[i, :2], axis=1).min()
                rel_times = arena.state_trajectory[:, 2] - arena.state_trajectory[0, 2]
                results.append(np.stack((distances, rel_times), axis=1))

        return results

    results = ray.get([distance.remote(problem) for problem in problem_factory.get_problem_list(10000)])

    def mod_time(time):
        shapes = [t.shape[0] for t in time]
        max_size = max(shapes)
        max_index = shapes.index(max_size)
        return np.stack([np.pad(t, ((0, max_size-t.shape[0]), (0,0)), 'constant', constant_values=np.nan) for t in time]), time[max_index][:, 1]

    lenght = len(results[0])

    for i in range(lenght):
        distances, times = mod_time([r[i] for r in results])
        distances = distances * units._METERS_PER_DEG_LAT_LON
        mean = np.nanmean(distances[:, :, 0], axis=0)
        std = np.nanstd(distances[:, :, 0], axis=0)
        error = 0.5 * std
        lower = mean - error
        upper = mean + error

        hours = times / 3600

        plt.plot(hours, mean, label=labels[i])
        plt.plot(hours, lower, color='tab:blue', alpha=0.1)
        plt.plot(hours, upper, color='tab:blue', alpha=0.1)
        plt.fill_between(hours, lower, upper, alpha=0.2)

    plt.legend()
    plt.xlabel('simulation time in hours')
    plt.ylabel('deviation in meters')
    plt.savefig('plots/7_setup/euler.png', dpi=300)
    plt.show()