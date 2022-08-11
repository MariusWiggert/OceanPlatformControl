from datetime import datetime
import ray.rllib.utils
import tempfile
from ray.rllib.agents.dqn.apex import ApexTrainer
import pickle
import time
import os
import json
import shutil
import numpy as np
from ray.tune.logger import UnifiedLogger
from pprint import pprint
from config.reinforcement_learning.apex_dqn_agent_config import apex_dqn_agent_config

from ocean_navigation_simulator.reinforcement_learning.OceanEnv import OceanEnv
from ocean_navigation_simulator.reinforcement_learning.RLRunner import RLRunner

script_start_time = time.time()
ray.init(
    'ray://localhost:10001',
    runtime_env={
        'working_dir': '.',
        'excludes': ['.git', './ocean_navigation_simulator'],
        'py_modules': ['ocean_navigation_simulator'],
    },
)
print(f"Code sent in {time.time()-script_start_time:.1f}s")


# %%
# SEED = 2022
#
# torch.manual_seed(SEED)
# torch.use_deterministic_algorithms(True)
# random.seed(SEED)
# np.random.seed(SEED)


def env_creator(env_config):
    print(env_config)
    return OceanEnv(config={
        'seed': np.random.randint(low=10000),
    })

ray.tune.registry.register_env("OceanEnv", env_creator)

runner = RLRunner(
    agent_class=ApexTrainer,
    agent_config=apex_dqn_agent_config,
    experiment_name='gulf_of_mexico/first_tries',
)

runner.agent.get_policy().model.base_model.summary()

print(f"Starting training ({runner.experiment_name}):")

runner.run(iterations=50)

print(f"Total Script Time: {time.time() - script_start_time:.2f}s = {(time.time() - script_start_time) / 60:.2f}min")
