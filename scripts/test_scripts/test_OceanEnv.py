import time
import datetime

from ocean_navigation_simulator.controllers.NaiveController import NaiveController
from ocean_navigation_simulator.reinforcement_learning.OceanEnv import OceanEnv


script_start = time.time()

env = OceanEnv(
    worker_index=1,
    config={
        "generation_folder": "/seaweed-storage/generation/increased_planner_area/",
        "scenario_name": "gulf_of_mexico_HYCOM_hindcast",
        "arena_steps_per_env_step": 1,
        "actions": 8,
        "render": True,
        "fake": False,  # one of: False, 'random', 'naive, 'hj_planner'
        "experiments_folder": "/seaweed-storage/tmp/",
        "feature_constructor_config": {
            "num_measurements": 0,
            "ttr": {
                "xy_width_degree": 0.4,
                "xy_width_points": 10,
                "normalize_at_curr_pos": True,
            },
        },
        "reward_function_config": {
            "target_bonus": 0,
            "fail_punishment": 0,
        },
    },
    verbose=2,
)
env.reset()

step = 1
done = False
rewards = []

print("")
print(
    f"Initial TTR in h: {env.hindcast_planner.interpolate_value_function_in_hours(observation=env.prev_obs):.2f}"
)
print(
    f"planner.current_data_t_0: {datetime.datetime.fromtimestamp(int(env.hindcast_planner.current_data_t_0), tz=datetime.timezone.utc)}"
)
print(
    f"planner.current_data_t_T: {datetime.datetime.fromtimestamp(int(env.hindcast_planner.current_data_t_T), tz=datetime.timezone.utc)}"
)
print(
    f"TTR at Ttarget: {env.hindcast_planner.interpolator((env.hindcast_planner.current_data_t_0 + env.hindcast_planner.reach_times, env.problem.end_region.lon.deg, env.problem.end_region.lat.deg))}"
)
print(f"{env.problem}")
print("")

while not done:
    action = env.hindcast_planner.get_action(env.prev_obs)

    start = time.time()
    features, reward, done, info = env.step(action)
    rewards.append(reward)
    print(f"Reward: {reward:.2f}")
    print(f"Total Reward: {sum(rewards):.2f}")
    print(
        f"TTR: {env.hindcast_planner.interpolate_value_function_in_hours(observation=env.prev_obs):.2f}"
    )
    # print(f'TTR @ Target: {env.hindcast_planner.interpolator((env.prev_obs.platform_state.date_time.timestamp(),env.problem.end_region.lon.deg,env.problem.end_region.lat.deg)):.2f}')
    print("")

    step += 1

print(f"Total Reward: {sum(rewards):.2f}")
print(
    f"Final TTR in : {env.hindcast_planner.interpolate_value_function_in_hours(observation=env.prev_obs):.2f}"
)
print(f"Passed Time in h: {env.problem.passed_seconds(env.arena.platform.state) / 3600:.2f}h")

print(f"### Mean Env Step Time: {(time.time()-script_start)/200:.3f}s")
print(f"### Script Time: {time.time()-script_start:.1f}s")
