from typing import Optional
import pandas as pd
import ray

from ocean_navigation_simulator.controllers.NaiveController import NaiveController
from ocean_navigation_simulator.reinforcement_learning.OceanEnv import OceanEnv
from ocean_navigation_simulator.scripts.Utils import Utils


class BaselineRunner:
    def __init__(
        self,
        generation_folder: str,
        ray_options: dict,
        verbose: Optional[int] = 0
    ):
        self.generation_folder = generation_folder
        self.verbose = verbose

        Utils.ensure_storage_connection()
        self.problems_df = pd.read_csv(f'{generation_folder}problems.csv').head(n=5)

        if verbose > 0:
            print(f'BaselineRunner: running baseline for {len(self.problems_df.shape[0])} problems')

        self.ray_results = ray.get([self.run_baseline.options(
            num_cpus=ray_options['resources']['CPU'],
            num_gpus=ray_options['resources']['GPU'],
            max_retries=ray_options['max_retries'],
            resources={i:ray_options['resources'][i] for i in ray_options['resources'] if i!='CPU' and i!= "GPU"}
        ).remote(
            index=index,
            row=row,
            verbose=self.verbose
        ) for index, row in self.problems_df.iterrows()])

        self.results_df = pd.DataFrame(self.ray_results).set_index('index').rename_axis(None)
        results_folder = f'/seaweed-storage/evaluation/{scenario_name}/{controller_class.__name__}_{time_string}'
        os.makedirs(results_folder , exist_ok=True)
        self.results_df.to_csv(f'{results_folder}/results.csv')

    @staticmethod
    @ray.remote(max_calls=1)
    def run_baseline(index, row, verbose):
        env = OceanEnv(
            verbose=verbose-1
        )
        env.reset()

        controller = NaiveController(problem=env.problem)

        step = 0
        done = False
        total_reward = 0

        # print(f'Initial TTR in h: {env.hindcast_planner.interpolate_value_function_in_hours_at_point(observation=env.prev_obs)}')

        while not done:
            action = controller.get_action(env.prev_obs)

            start = time.time()
            features, reward, done, info = env.step(action)
            # print(f'OceanEnv Step {step} ({time.time()-start:.1f}s)')
            # print(features.shape, reward, done, info)
            total_reward += reward
            # print(f'Current TTR in h at Step {step}: {env.hindcast_planner.interpolate_value_function_in_hours_at_point(observation=env.prev_obs):.2f}')
            # print(f'Total Reward (Improvement in TTR in h): {total_reward:.2f}')

            step += 1

        # print(f'Final TTR in : {env.hindcast_planner.interpolate_value_function_in_hours_at_point(observation=env.prev_obs)}')
        # print(f'Passed Time in h: {env.problem.passed_seconds(env.arena.platform.state) / 3600:.2f}h')

