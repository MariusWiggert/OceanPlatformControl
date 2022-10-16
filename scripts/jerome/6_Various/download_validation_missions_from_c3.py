from tqdm import tqdm
import pandas as pd

import ocean_navigation_simulator.utils.misc
import ocean_navigation_simulator.utils.paths as paths
from ocean_navigation_simulator.reinforcement_learning_scripts.Utils import Utils

c3 = ocean_navigation_simulator.utils.misc.get_c3()

#%%
missions = c3.Mission.fetch(spec={'filter': 'experiment.id=="Short_Horizon_CopernicusGT"'})
print(f'Total Missions: {missions.count}')

simConfig = c3.Experiment.get("Short_Horizon_CopernicusGT").simConfig

missions_export = []
for mission in tqdm(missions.objs):
    sim_runs = c3.OceanSimRun.fetch(spec={'filter': f'mission.id=="{mission.id}"'})

    if mission.status == 'ready_to_run' and sim_runs.count != 6:
        print('Experiments not correctly run!')

    mission_export = {
        'name': mission.id,
        't_0': mission.x_0.t,
        'x_0_lon': mission.x_0.x.longitude,
        'x_0_lat': mission.x_0.x.latitude,
        'x_T_lon': mission.x_T.longitude,
        'x_T_lat': mission.x_T.latitude,
        'target_radius': mission.x_T_radius,
        'timeout_in_h': simConfig.T_in_h,
        'feasible': mission.status == 'ready_to_run',
        'Naive_success': None,
        'Naive_time': None,
        'HJ_success': None,
        'HJ_time': None,
    }

    if sim_runs.count > 0:
        for sim_run in sim_runs.objs:
            if 'StraightLine' in sim_run.controllerSetting.id:
                mission_export['Naive_success'] = sim_run.terminationReason == 'goal_reached'
                mission_export['Naive_time'] = sim_run.T_arrival_time
            if 'Multi_Reach_Back' in sim_run.controllerSetting.id:
                mission_export['HJ_success'] = sim_run.terminationReason == 'goal_reached'
                mission_export['HJ_time'] = sim_run.T_arrival_time

    missions_export.append(mission_export)

# Save Missions as csv
all_missions = pd.DataFrame(missions_export)
all_missions.to_csv(paths.DATA + 'missions/all.csv')
feasible_missions = all_missions[all_missions['feasible']]
feasible_missions.to_csv(paths.DATA + 'missions/feasible.csv')
failed_missions = all_missions[all_missions['HJ_success'] == False]
failed_missions.to_csv(paths.DATA + 'missions/HJ_failed.csv')

# Checking Correctness: Mean Success Rate should be the same as in "Navigating Underactuated Agents by Hitchhiking Forecast Flows"
print(f'Feasible: {feasible_missions.shape[0]}/{all_missions.shape[0]}')
print(f'Naive Average Success: {feasible_missions["Naive_success"].mean():.1%} of Feasible Missions')
print(f'HJ Average Success: {feasible_missions["HJ_success"].mean():.1%} of Feasible Missions')