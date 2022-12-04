from ocean_navigation_simulator.reinforcement_learning.runners.TrainingRunner import (
    TrainingRunner,
)
from ocean_navigation_simulator.utils import cluster_utils

cluster_utils.ensure_storage_connection()

TrainingRunner.clean_results(
    folder="/seaweed-storage/experiments/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/",
    iteration_limit=10,
    delete=False,
    ignore_errors=False,
    verbose=1,
)
