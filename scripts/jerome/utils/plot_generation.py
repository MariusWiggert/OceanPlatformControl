import os

import pandas as pd
from matplotlib import pyplot as plt

from ocean_navigation_simulator.reinforcement_learning.runners.GenerationRunner import (
    GenerationRunner,
)

# import os
# for dir in os.listdir(
#     "/seaweed-storage/generation/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/"
# ):
#     GenerationRunner.plot_starts_and_targets(
#         "/seaweed-storage/generation/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/"
#         + dir
#         + "/",
#     )
#     GenerationRunner.plot_target_dates_histogram(
#         "/seaweed-storage/generation/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/"
#         + dir
#         + "/",
#     )
from ocean_navigation_simulator.utils import cluster_utils

results_folder = "/seaweed-storage/generation/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/divers_training_2022_10_21_04_10_04/"

# GenerationRunner.plot_starts_and_targets(results_folder)
# GenerationRunner.plot_target_dates_histogram(results_folder)

# Step 1: Load Data
print('Storage Connection:', cluster_utils.check_storage_connection())
problems_df = pd.read_csv(f"{results_folder}problems.csv")
analysis_folder = f"{results_folder}analysis/"
os.makedirs(analysis_folder, exist_ok=True)

problems_df[problems_df['random'] == False].plot.hist('ttr_in_h', bins=100)

plt.show()

