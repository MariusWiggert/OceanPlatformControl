from ocean_navigation_simulator.reinforcement_learning.missions.FileProblemFactory import (
    FileProblemFactory,
)
from ocean_navigation_simulator.reinforcement_learning.runners.EvaluationRunner import (
    EvaluationRunner,
)


# folder = "/seaweed-storage/evaluation/Random/"
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
        "start": 70000,
        "stop": 10000,
    },
).indices
# #
# print(len(indexes))
#
# hj = folder + "feasible.csv"

EvaluationRunner.print_results(
    # csv_file="/seaweed-storage/experiments/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/dense no_meta_2022_11_29_11_07_43/eval_gp_std_cp70_2022_11_04_13_30_45.csv",
    csv_file="/seaweed-storage/evaluation/CachedHJReach2DPlannerForecast/feasible.csv",
    indexes=indexes,
)
