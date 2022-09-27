import ray.rllib.utils
from ray.rllib.agents.ppo import PPOTrainer
import time
import numpy as np
from ray.tune.logger import UnifiedLogger

from ocean_navigation_simulator.reinforcement_learning.DoubleGyreEnv import DoubleGyreEnv
from config.reinforcement_learning.ppo_agent_config import ppo_agent_config
from ocean_navigation_simulator.reinforcement_learning.scripts.RLRunner import RLRunner

script_start_time = time.time()

ray.tune.registry.register_env(
    "DoubleGyreEnv", lambda env_config: DoubleGyreEnv(config={"seed": np.random.randint(low=10000)})
)

runner = RLRunner(
    agent_class=PPOTrainer,
    agent_config=ppo_agent_config,
    experiment_name="double_gyre/angle_feature_simple_nn_with_currents",
    verbose=10,
)


print(f"Starting training ({runner.experiment_name}):")

runner.run(iterations=400)

runner.agent.get_policy().model.base_model.summary()
print(
    f"Total Script Time: {time.time() - script_start_time:.2f}s = {(time.time() - script_start_time) / 60:.2f}min"
)
