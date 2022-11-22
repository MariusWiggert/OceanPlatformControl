import pandas as pd

# import wandb
#
# from ocean_navigation_simulator.reinforcement_learning.missions.FileProblemFactory import (
#     FileProblemFactory,
# )
from ocean_navigation_simulator.reinforcement_learning.missions.FileProblemFactory import (
    FileProblemFactory,
)
from ocean_navigation_simulator.reinforcement_learning.runners.EvaluationRunner import (
    EvaluationRunner,
)


folder = "/seaweed-storage/evaluation/Random/"
# files = ['eval_feasible_40000m_2022_11_17_07_32_19.csv', 'eval_feasible_30000m_2022_11_17_01_19_11.csv', 'eval_feasible_9461m_2022_11_17_00_31_55.csv']
#
# df = pd.concat([pd.read_csv(folder + f) for f in files])
# df = df.drop(columns='Unnamed: 0')
#
# df.to_csv(folder + 'combined.csv')

indexes = FileProblemFactory(
    csv_file="/seaweed-storage/generation/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/divers_training_improved_2022_10_23_05_10_12/problems.csv",
    filter={
        "no_random": True,
        "start": 70204,
        "stop": 10000,
    },
).indices
#
print(len(indexes))

hj = folder + "feasible.csv"

EvaluationRunner.print_results(
    # csv_file="/seaweed-storage/experiments/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/gp_error_and_std_half_hour_2022_11_03_13_55_30/eval_gp_std_cp70_2022_11_04_13_30_45.csv",
    csv_file="/seaweed-storage/evaluation/CachedHJReach2DPlannerHindcast/eval_9461m_2022_11_14_03_22_58.csv",
    indexes=indexes,
)

# EvaluationRunner.update_wandb_summary(
#     csv_file=folder + "feasible.csv",
#     wandb_run_id="1pldfcdo",
#     time_string="2022_11_04_14_50_00",
#     indexes=indexes,
# )
#
# wandb.init(
#     project="seaweed-rl",
#     entity="jeromejeannin",
#     dir="/seaweed-storage/",
#     id=wandb_run_id,
#     resume="must",
# )
#
# wandb.run.summary[f"ray/tune/evaluation/custom_metrics/success_mean"] = df["success"].mean()
# wandb.run.summary[f"ray/tune/evaluation/custom_metrics/arrival_time_in_h_mean"] = (
#     df[df["success"]]["running_time"].mean() / 3600
# )
#
# wandb.run.summary[f"validation_204"] = {
#     "date": time_string,
#     "length": df.shape[0],
#     "success_mean": df["success"].mean(),
#     "running_time": df["running_time"].mean(),
# }
# wandb.finish()
