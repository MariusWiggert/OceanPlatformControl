import time
import ray
import matplotlib.pyplot as plt

from ocean_navigation_simulator.reinforcement_learning.OceanEnv import OceanEnv
from ocean_navigation_simulator.reinforcement_learning_scripts.Utils import Utils

script_start = time.time()

@ray.remote
def test_environment(worker_index, higher_resolution) -> OceanEnv:
    env = OceanEnv(
        config={
            'generation_folder': '/seaweed-storage/generation/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/fixed_forecast_50000_batches/',
            'scenario_name': 'gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast',
            'scenario_config': {
                'platform_dict': {
                    'dt_in_s': 60.0 if higher_resolution else 600.0
                }
            },
            'arena_steps_per_env_step': 10 if higher_resolution else 1,
            'actions': 8,
            'render': False,
            'fake': 'hj_planner_hindcast',  # one of: False, 'random', 'naive, 'hj_planner'
            'experiment_folder': '/seaweed-storage/tmp/',
        },
        feature_constructor_config={
            'num_measurements': 0,
            'map': {
                'xy_width_degree': 0.2,
                'xy_width_points': 5,
                'flatten': False,
                'features': {
                    'ttr_forecast': False,
                    'ttr_hindcast': True,
                    'observer': [],
                }
            },
        },
        reward_function_config={
            'delta_ttr_forecast': 0,
            'delta_ttr_hindcast': 1,
            'target_bonus': 0,
            'fail_punishment': 0,
        },
        worker_index=worker_index,
        verbose=2,
    )
    env.reset()

    done = False

    while not done:
        _, _, done, _ = env.step(action=1)

    return env


Utils.ray_init()


envs = ray.get([test_environment.remote(worker_index=1, higher_resolution=False), test_environment.remote(worker_index=1, higher_resolution=True)])

fig = plt.figure()
plt.suptitle('TTR improvement')
plt.title('10min Step Time, HJ Hindcast Planner', fontsize=10)

for env in envs[:7]:
    p = plt.plot(env.rewards, label=f'Problem[Group {env.problem.extra_info["group"]} Batch {env.problem.extra_info["batch"]} Index {env.problem.extra_info["factory_index"]}], Mean: {sum(env.rewards)/len(env.rewards):.4f}')
    plt.hlines(sum(env.rewards)/len(env.rewards),0,len(env.rewards)-1,linestyle='--',alpha=0.7,color=p[0].get_color())

plt.legend(prop={'size': 8})
plt.show()

# print(f'Total Reward: {sum(rewards):.4f}')
# print(f'Final TTR n h: {env.hindcast_planner.interpolate_value_function_in_hours(point=env.prev_obs.platform_state.to_spatio_temporal_point()):.4f}')
# print(f'Passed Time: {env.problem.passed_seconds(env.arena.platform.state) / 3600:.4f}h')
#
# print(f'### Mean Env Step Time: {(time.time()-script_start)/step:.3f}s')
# print(f'### Script Time: {time.time()-script_start:.1f}s')