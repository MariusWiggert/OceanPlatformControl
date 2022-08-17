import time
from ray.rllib.agents.dqn.apex import ApexTrainer

from ocean_navigation_simulator.scripts.RLRunner import RLRunner
from ocean_navigation_simulator.scripts.RayUtils import RayUtils

print('Script started ...')
script_start_time = time.time()

RayUtils.init_ray()
RayUtils.check_storage_connection()

runner = RLRunner(
    name='first_tries',
    agent_class=ApexTrainer,
    agent_config={},
    feature_constructor_config={},
    verbose=10
)
# runner.agent.get_policy().model.summary()
runner.run(iterations=10)


script_time = time.time()-script_start_time
print(f"Script finished in {script_time/60:.0f}min {script_time%60:.0f}s.")