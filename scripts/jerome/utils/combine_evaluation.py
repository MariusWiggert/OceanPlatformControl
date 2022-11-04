# import pandas as pd
# import wandb

from ocean_navigation_simulator.reinforcement_learning.missions.FileProblemFactory import (
    FileProblemFactory,
)

# from ocean_navigation_simulator.reinforcement_learning.runners.EvaluationRunner import (
#     EvaluationRunner,
# )

# folder = "/seaweed-storage/evaluation/CachedHJReach2DPlannerForecast/"
# files = ['evaluation_2022_10_27_01_27_24.csv', 'evaluation_2022_10_27_17_37_01.csv', 'evaluation_2022_10_27_23_26_41.csv']

# df = pd.concat([pd.read_csv(folder + f) for f in files])
# df = df.drop(columns='Unnamed: 0')

# df.to_csv(folder + 'combined.csv')

indexes = FileProblemFactory(
    csv_file="/seaweed-storage/generation/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/divers_training_improved_2022_10_23_05_10_12/problems.csv",
    filter={
        "no_random": True,
        "start": 70204,
        "stop": 5100,
    },
).indices

print(len(indexes))

# df = pd.read_csv(folder + "combined.csv", index_col=0)
# df = df[df['index'].isin(indexes)]
# print("success:", df["success"].mean())
# print("running_time:", df["running_time"].mean())
#
#
# csv_file = folder + "combined.csv"
# wandb_run_id = "1pldfcdo"
# time_string = "2022_10_27_23_26_41"
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
