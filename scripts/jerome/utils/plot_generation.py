import os

import pandas as pd
from matplotlib import pyplot as plt

from ocean_navigation_simulator.reinforcement_learning.runners.GenerationRunner import (
    GenerationRunner,
)

for dir in os.listdir(
    "/seaweed-storage/generation/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/"
):
    f = "/seaweed-storage/generation/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/" + dir + "/"
    # GenerationRunner.plot_starts_and_targets(f)
    # GenerationRunner.plot_target_dates_histogram(f)
    GenerationRunner.plot_ttr_histogram(f)


from ocean_navigation_simulator.utils import cluster_utils

# generation = "/seaweed-storage/generation/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/divers_training_improved_2022_10_23_05_10_12/"
# generation = "~/Desktop/"

# GenerationRunner.plot_starts_and_targets(generation)
# GenerationRunner.plot_target_dates_histogram(generation)
