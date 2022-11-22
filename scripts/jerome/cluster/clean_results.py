from ocean_navigation_simulator.reinforcement_learning.runners.TrainingRunner import (
    TrainingRunner,
)

TrainingRunner.clean_results(
    folder="/seaweed-storage/experiments/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/",
    iteration_limit=10,
    delete=False,
    verbose=1,
)
