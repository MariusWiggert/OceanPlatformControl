import datetime
import json
import pickle
import shutil
import time
from pprint import pprint
import os
from ray.tune.logger import UnifiedLogger
from ray.rllib.agents.dqn.apex import ApexTrainer


class RLRunner:
    def __init__(self, agent_class, agent_config, experiment_name):
        self.agent_class = agent_class
        self.agent_config = agent_config

        self.time_string = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        self.experiment_name = experiment_name + '_' + self.time_string
        self.experiment_path = 'experiments/' + experiment_name + '/'
        self.checkpoints_path = self.experiment_path + 'checkpoints/'
        self.plots_path = self.experiment_path + 'plots/'

        # Step 1: Create or Empty experiment folder
        if os.path.exists(self.experiment_path):
            shutil.rmtree(self.experiment_path)
        os.makedirs(self.checkpoints_path , exist_ok=True)
        os.makedirs(self.plots_path, exist_ok=True)

        # Step 2: Save configuration
        pickle.dump(self.agent_config, open(self.experiment_path + 'config.p', "wb"))
        json.dump(self.agent_config, open(self.experiment_path + '/config.json', "w"))

        # Step 3: Ray results Directory
        ray_results_dir = f"{os.path.expanduser('~/ray_results')}/{self.experiment_name.replace('/', '_')}"
        os.makedirs(ray_results_dir, exist_ok=True)

        # Step 4: Create Agent
        self.agent = ApexTrainer(agent_config, logger_creator=lambda config: UnifiedLogger(config, ray_results_dir, loggers=None))
        self.results = []

    def run(self, iterations = 100, verbose=True):
        for iteration in range(1, iterations + 1):
            start = time.time()
            result = self.agent.train()
            iteration_time = time.time() - start

            self.results.append(result)

            if verbose:
                self.print_result(iteration, iterations, result, iteration_time)

            pickle.dump(self.result, open(self.experiment_path+'/results.p', "wb"))
            json.dump(self.results, open(self.experiment_path+'/results.json', "w"))

    def print_result(self, iteration, iterations, result, iteration_time):
        print(' ')
        print(' ')
        print(f'--------- Iteration {iteration} (total samples {result["info"]["num_env_steps_trained"]}) ---------')

        print('-- Episode Rewards --')
        print(f'[{", ".join([f"{elem:.1f}" for elem in result["hist_stats"]["episode_reward"][-min(25, result["episodes_this_iter"]):]])}]')
        print(f'Mean: {result["episode_reward_mean"]:.2f}')
        print(f'Max:  {result["episode_reward_max"]:.2f},')
        print(f'Min:  {result["episode_reward_min"]:.2f}')
        print(' ')

        print('-- Episode Length --')
        episodes_this_iteration = result["hist_stats"]["episode_lengths"][-result["episodes_this_iter"]:]
        print(result["hist_stats"]["episode_lengths"][-min(40, result["episodes_this_iter"]):])
        print(f'Mean: {result["episode_len_mean"]:.2f}')
        print(f'Min:  {min(episodes_this_iteration):.2f}')
        print(f'Max:  {max(episodes_this_iteration):.2f}')
        print(f'Number of Episodes: {len(episodes_this_iteration)}')
        print(f'Sum Episode Steps:  {sum(episodes_this_iteration)}')
        print(f'Samples for Training: {result["num_env_steps_trained_this_iter"]}')
        print(' ')

        print('-- Timing --')
        pprint(result["sampler_perf"])
        print(f'total time per step: {sum(result["sampler_perf"].values()):.2f}ms')
        print(f'iteration time: {iteration_time:.2f}s ({iterations * (iteration_time) / 60:.1f}min for {iterations} iterations, {(iterations - iteration) * (iteration_time) / 60:.1f}min to go)')