from datetime import datetime
import ray.rllib.utils
import tempfile
from ray.rllib.agents.ppo import PPOTrainer
import pickle
import time
import os
import json
import shutil
import torch
import random
import numpy as np
from ray.tune.logger import UnifiedLogger
from pprint import pprint

from ocean_navigation_simulator.reinforcement_learning.DoubleGyreEnv import DoubleGyreEnv

script_start_time = time.time()

ray_cluster = ray.init(
    'ray://localhost:10001',
    runtime_env={
        'working_dir': '.',
        'excludes': ['data', 'generated_media', 'hj_reachability', 'models', '.git', 'ocean_navigation_simulator', 'results'],
        'py_modules': ['ocean_navigation_simulator'],
        # "env_vars": {"TF_WARNINGS": "none"}
    },
)
print(f"Code sent in {time.time()-script_start_time:.1f}s")


# %%
SEED = 2022

torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
random.seed(SEED)
np.random.seed(SEED)


def env_creator(env_config):
    return DoubleGyreEnv(config={
        'seed': np.random.randint(low=10000),
        'arena_steps_per_env_step': 1,
    })

ray.tune.registry.register_env("PlatformEnv", env_creator)

config = {
    # ========= Environment Settings =========
    # The environment specifier:
    # This can either be a tune-registered env, via
    # `tune.register_env([name], lambda env_ctx: [env object])`,
    # or a string specifier of an RLlib supported type. In the latter case,
    # RLlib will try to interpret the specifier as either an openAI gym env,
    # a PyBullet env, a ViZDoomGym env, or a fully qualified classpath to an
    # Env class, e.g. "ray.rllib.examples.env.random_env.RandomEnv".
    "env": "PlatformEnv",

    # Number of steps after which the episode is forced to terminate. Defaults
    # to `env.spec.max_episode_steps` (if present) for Gym envs.
    "horizon": 100,
    # Calculate rewards but don't reset the environment when the horizon is
    # hit. This allows value estimation and RNN state to span across logical
    # episodes denoted by horizon. This only has an effect if horizon != inf.
    "soft_horizon": True,
    # Don't set 'done' at the end of the episode.
    # In combination with `soft_horizon`, this works as follows:
    # - no_done_at_end=False soft_horizon=False:
    #   Reset env and add `done=True` at end of each episode.
    # - no_done_at_end=True soft_horizon=False:
    #   Reset env, but do NOT add `done=True` at end of the episode.
    # - no_done_at_end=False soft_horizon=True:
    #   Do NOT reset env at horizon, but add `done=True` at the horizon
    #   (pretending the episode has terminated).
    # - no_done_at_end=True soft_horizon=True:
    #   Do NOT reset env at horizon and do NOT add `done=True` at the horizon.
    "no_done_at_end": False,

    # Arguments dict passed to the env creator as an EnvContext object (which
    # is a dict plus the properties: num_workers, worker_index, vector_index,
    # and remote).
    # "env_config": {
    #     'seed': 2022,
    #     'arena_steps_per_env_step': 1,
    # },

    # Whether to use "rllib" or "deepmind" preprocessors by default
    # Set to None for using no preprocessor. In this case, the model will have
    # to handle possibly complex observations from the environment.
    "preprocessor_pref": 'rllib',

    # ========= Settings for Rollout Worker processes =========
    # Use 1 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 38,
    # "num_envs_per_worker": 1,
    # "remote_worker_envs": True,
    # "remote_env_batch_wait_ms": 0,
    "num_gpus": 1,

    # Divide episodes into fragments of this many steps each during rollouts.
    # Sample batches of this size are collected from rollout workers and
    # combined into a larger batch of `train_batch_size` for learning.
    #
    # For example, given rollout_fragment_length=100 and train_batch_size=1000:
    #   1. RLlib collects 10 fragments of 100 steps each from rollout workers.
    #   2. These fragments are concatenated and we perform an epoch of SGD.
    #
    # When using multiple envs per worker, the fragment size is multiplied by
    # `num_envs_per_worker`. This is since we are collecting steps from
    # multiple envs in parallel. For example, if num_envs_per_worker=5, then
    # rollout workers will return experiences in chunks of 5*100 = 500 steps.
    #
    # The dataflow here can vary per algorithm. For example, PPO further
    # divides the train batch into minibatches for multi-epoch SGD.
    "rollout_fragment_length": 200,
    # How to build per-Sampler (RolloutWorker) batches, which are then
    # usually concat'd to form the train batch. Note that "steps" below can
    # mean different things (either env- or agent-steps) and depends on the
    # `count_steps_by` (multiagent) setting below.
    # truncate_episodes: Each produced batch (when calling
    #   RolloutWorker.sample()) will contain exactly `rollout_fragment_length`
    #   steps. This mode guarantees evenly sized batches, but increases
    #   variance as the future return must now be estimated at truncation
    #   boundaries.
    # complete_episodes: Each unroll happens exactly over one episode, from
    #   beginning to end. Data collection will not stop unless the episode
    #   terminates or a configured horizon (hard or soft) is hit.
    "batch_mode": "truncate_episodes",

    # ========= Debug Settings =========
    # # Set the ray.rllib.* log level for the agent process and its workers.
    # # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
    # # periodically print out summaries of relevant internal dataflow (this is
    # # also printed out once at startup at the INFO level). When using the
    # # `rllib train` command, you can also use the `-v` and `-vv` flags as
    # # shorthand for INFO and DEBUG.
    # "log_level": "WARN",
    # # Callbacks that will be run during various phases of training. See the
    # # `DefaultCallbacks` class and `examples/custom_metrics_and_callbacks.py`
    # # for more usage information.
    # # "callbacks": DefaultCallbacks,
    # # Whether to attempt to continue training if a worker crashes. The number
    # # of currently healthy workers is reported as the "num_healthy_workers"
    # # metric.
    # "ignore_worker_failures": False,
    # # Log system resource metrics to results. This requires `psutil` to be
    # # installed for sys stats, and `gputil` for GPU metrics.
    # "log_sys_usage": True,
    # # Use fake (infinite speed) sampler. For testing only.
    # "fake_sampler": False,


    # ========= Debug Settings =========
    # Set the ray.rllib.* log level for the agent process and its workers.
    # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
    # periodically print out summaries of relevant internal dataflow (this is
    # also printed out once at startup at the INFO level). When using the
    # `rllib train` command, you can also use the `-v` and `-vv` flags as
    # shorthand for INFO and DEBUG.
    "log_level": "WARN",


    # ========= Deep Learning Framework Settings =========
    # tf: TensorFlow (static-graph)
    # tf2: TensorFlow 2.x (eager or traced, if eager_tracing=True)
    # tfe: TensorFlow eager (or traced, if eager_tracing=True)
    # torch: PyTorch
    "framework": "tf2",
    # # Enable tracing in eager mode. This greatly improves performance
    # # (speedup ~2x), but makes it slightly harder to debug since Python
    # # code won't be evaluated after the initial eager pass.
    # # Only possible if framework=[tf2|tfe].
    # "eager_tracing": False,
    # # Maximum number of tf.function re-traces before a runtime error is raised.
    # # This is to prevent unnoticed retraces of methods inside the
    # # `..._eager_traced` Policy, which could slow down execution by a
    # # factor of 4, without the user noticing what the root cause for this
    # # slowdown could be.
    # # Only necessary for framework=[tf2|tfe].
    # # Set to None to ignore the re-trace count and never throw an error.
    # "eager_max_retraces": 20,

    # ========= PPO-specific Settings =========
    # Training batch size, if applicable. Should be >= rollout_fragment_length.
    # Samples batches will be concatenated together to a batch of this size,
    # which is then passed to SGD.
    "train_batch_size": 4000,

    # Total SGD batch size across all devices for SGD. This defines the
    # minibatch size within each epoch.
    "sgd_minibatch_size": 128,
    # Whether to shuffle sequences in the batch when training (recommended).
    "shuffle_sequences": True,
    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    "num_sgd_iter": 30,

    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    "model": {
        # "fcnet_hiddens": [16, 64, 16, 6, 8, 8],
        "fcnet_hiddens": [32, 32],
        "fcnet_activation": "relu",
        "vf_share_layers": False,
    },

    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    # "vf_clip_param": 10.0,

    # ========= Advanced Rollout Settings =========
    # This argument, in conjunction with worker_index, sets the random seed of
    # each worker, so that identically configured trials will have identical
    # results. This makes experiments reproducible.
    "seed": SEED,

    # ========= API deprecations/simplifications/changes =========
    'disable_env_checking': True,

    # # ========= Evaluation Settings =========
    # # Evaluate with every `evaluation_interval` training iterations.
    # # The evaluation stats will be reported under the "evaluation" metric key.
    # # Note that for Ape-X metrics are already only reported for the lowest
    # # epsilon workers (least random workers).
    # # Set to None (or 0) for no evaluation.
    # "evaluation_interval": 5,
    # # Duration for which to run evaluation each `evaluation_interval`.
    # # The unit for the duration can be set via `evaluation_duration_unit` to
    # # either "episodes" (default) or "timesteps".
    # # If using multiple evaluation workers (evaluation_num_workers > 1),
    # # the load to run will be split amongst these.
    # # If the value is "auto":
    # # - For `evaluation_parallel_to_training=True`: Will run as many
    # #   episodes/timesteps that fit into the (parallel) training step.
    # # - For `evaluation_parallel_to_training=False`: Error.
    # "evaluation_duration": 10,
    # # The unit, with which to count the evaluation duration. Either "episodes"
    # # (default) or "timesteps".
    # "evaluation_duration_unit": "episodes",
    # # Whether to run evaluation in parallel to a Trainer.train() call
    # # using threading. Default=False.
    # # E.g. evaluation_interval=2 -> For every other training iteration,
    # # the Trainer.train() and Trainer.evaluate() calls run in parallel.
    # # Note: This is experimental. Possible pitfalls could be race conditions
    # # for weight synching at the beginning of the evaluation loop.
    # "evaluation_parallel_to_training": False,
    # # Internal flag that is set to True for evaluation workers.
    # "in_evaluation": False,
    # # Typical usage is to pass extra args to evaluation env creator
    # # and to disable exploration by computing deterministic actions.
    # # IMPORTANT NOTE: Policy gradient algorithms are able to find the optimal
    # # policy, even if this is a stochastic one. Setting "explore=False" here
    # # will result in the evaluation workers not using this optimal policy!
    # "evaluation_config": {
    #     # Example: overriding env_config, exploration, etc:
    #     # "env_config": {...},
    #     # "explore": False
    # },
    # # Number of parallel workers to use for evaluation. Note that this is set
    # # to zero by default, which means evaluation will be run in the trainer
    # # process (only if evaluation_interval is not None). If you increase this,
    # # it will increase the Ray resource usage of the trainer since evaluation
    # # workers are created separately from rollout workers (used to sample data
    # # for training).
    # "evaluation_num_workers": 1,
    # # Customize the evaluation method. This must be a function of signature
    # # (trainer: Trainer, eval_workers: WorkerSet) -> metrics: dict. See the
    # # Trainer.evaluate() method to see the default implementation.
    # # The Trainer guarantees all eval workers have the latest policy state
    # # before this function is called.
    # "custom_eval_function": None,
    # # Make sure the latest available evaluation results are always attached to
    # # a step result dict.
    # # This may be useful if Tune or some other meta controller needs access
    # # to evaluation metrics all the time.
    # "always_attach_evaluation_results": False,
    # # Store raw custom metrics without calculating max, min, mean
    # "keep_per_episode_custom_metrics": False,

    # ========= Exploration Settings =========
    # Default exploration behavior, iff `explore`=None is passed into
    # compute_action(s).
    # Set to False for no exploration behavior (e.g., for evaluation).
    "explore": True,
    # Provide a dict specifying the Exploration object's config.
    # "exploration_config": {
    # The Exploration class to use. In the simplest case, this is the name
    # (str) of any class present in the `rllib.utils.exploration` package.
    # You can also provide the python class directly or the full location
    # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
    # EpsilonGreedy").
    # "type": "GaussianNoise",
    # "type": "Epsilon",
    # Add constructor kwargs here (if any).
    # "random_timesteps": 1000,
    # "stddev": 0.1,
    # "initial_scale": 1,
    # "final_scale": 0.02,
    # "scale_timesteps": 1e5,
    # },

}

results = []
model_name = 'angle_feature_simple_nn_with_currents'
model_path = 'models/simplified_double_gyre/' + model_name + '/'
check_point_path = model_path + 'checkpoints/'
if not os.path.exists('models'):
    os.mkdir('models')
if not os.path.exists('models/double_gyre'):
    os.mkdir('models/double_gyre')
if os.path.exists(model_path):
    shutil.rmtree(model_path)
os.mkdir(model_path)
os.mkdir(check_point_path)
pickle.dump(config, open(model_path + 'config.p', "wb"))
with open(model_path + '/config.json', "w") as outfile:
    json.dump(config, outfile)

timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
logdir_prefix = "{}_{}_".format(model_name, timestr)
DEFAULT_RESULTS_DIR = os.path.expanduser("~/ray_results")
if not os.path.exists(DEFAULT_RESULTS_DIR):
    os.makedirs(DEFAULT_RESULTS_DIR)
logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=DEFAULT_RESULTS_DIR)


def logger_creator(config):
    return UnifiedLogger(config, logdir, loggers=None)

agent = PPOTrainer(config=config, logger_creator=logger_creator)

model = agent.get_policy().model.base_model.summary()

print(f"starting training ({model_name}):")

ITERATIONS = 400
for i in range(1, ITERATIONS + 1):
    inter = time.time()

    result = agent.train()
    results.append(result)

    print(' ')
    print(' ')
    print(f'--------- Iteration {i} (total samples {result["info"]["num_env_steps_trained"]}) ---------')

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
    print(f'iteration time: {time.time() - inter:.2f}s ({ITERATIONS * (time.time() - inter) / 60:.1f}min for {ITERATIONS} iterations, {(ITERATIONS - i) * (time.time() - inter) / 60:.1f}min to go)')

    agent.save(check_point_path)

    # pickle.dump(result, open(check_point_path+f'checkpoint_{i:06d}/results.p', "wb"))
    # with open(check_point_path+f'checkpoint_{i:06d}/results.json', "w") as outfile:
    #     json.dump(results, outfile)

# pickle.dump(result, open(model_path+'/results.p', "wb"))
# with open(model_path+'/results.json', "w") as outfile:
#     json.dump(results, outfile)

print(f"Total Script Time: {time.time() - script_start_time:.2f}s = {(time.time() - script_start_time) / 60:.2f}min")
