from ocean_navigation_simulator.reinforcement_learning.missions.FileProblemFactory import (
    FileProblemFactory,
)
from ocean_navigation_simulator.reinforcement_learning.runners.EvaluationRunner import (
    EvaluationRunner,
)

baseline = {
    'random': "/seaweed-storage/evaluation/Random/feasible.csv",
    'naive': "/seaweed-storage/evaluation/Naive/feasible.csv",
    'hj fc': "/seaweed-storage/evaluation/CachedHJReach2DPlannerForecast/feasible.csv",
    'hj hc': "/seaweed-storage/evaluation/CachedHJReach2DPlannerHindcast/eval_9461m_2022_11_14_03_22_58.csv",
}

experiments = [
    '/seaweed-storage/experiments/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/grouped_cnn second_fc_2022_11_20_03_54_18/eval_cp50_2000m_2022_11_21_10_48_17.csv',
    '/seaweed-storage/experiments/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/grouped_cnn second_fc_2022_11_20_03_54_18/eval_cp100_2000m_2022_11_21_08_40_24.csv',
    '/seaweed-storage/experiments/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/grouped_cnn second_fc_2022_11_20_03_54_18/eval_cp150_2000m_2022_11_21_07_02_33.csv',
]

for csv_file in experiments:
    print(csv_file)

    for start, stop in [(70000, 204), (70204, 1000), (70204, 2000), (70204, 10000), (0, False)]:
        indexes = FileProblemFactory(
            csv_file="/seaweed-storage/generation/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/divers_training_improved_2022_10_23_05_10_12/problems.csv",
            filter={
                "no_random": True,
                "start": start,
                "stop": stop,
            },
        ).indices
        EvaluationRunner.print_results(
            csv_file=csv_file,
            indexes=indexes,
        )
        print('')
    print('')