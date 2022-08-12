import time
from ray.rllib.agents.dqn.apex import ApexTrainer

from config.reinforcement_learning.apex_dqn_agent_config import apex_dqn_agent_config
from ocean_navigation_simulator.reinforcement_learning.RLRunner import RLRunner

script_start_time = time.time()

runner = RLRunner(
    agent_class=ApexTrainer,
    agent_config=apex_dqn_agent_config,
    experiment_name='gulf_of_mexico/first_tries',
    verbose=True
)

runner.agent.get_policy().model.base_model.summary()

print(f"Starting training ({runner.experiment_name}):")

runner.run(iterations=2)

print(f"Total Script Time: {time.time() - script_start_time:.2f}s = {(time.time() - script_start_time) / 60:.2f}min")