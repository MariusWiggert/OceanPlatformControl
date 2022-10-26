from ocean_navigation_simulator.reinforcement_learning.runners.GenerationRunner import (
    GenerationRunner,
)
from ocean_navigation_simulator.utils import cluster_utils

cluster_utils.init_ray()
cluster_utils.ensure_storage_connection()

results_folder = "/seaweed-storage/generation/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/divers_training_improved_2022_10_23_05_10_12/"


# Step 1: Analyze Missing Files
# existing, missing = GenerationRunner.analyze_batches(generation, 10000)


# Step 2: Rerun missing Batches
# import pickle
# with open(f"{generation}config/config.pickle", "rb") as f:
#     config = pickle.load(f)
# GenerationRunner.rerun_missing_batches(
#     config=config,
#     generation=generation,
#     length=10000,
#     verbose=3,
# )

# Step 3: Combine Batches
GenerationRunner.combine_batches(
    results_folder=results_folder,
    length=10000,
)

# Step 4: Analyse Generation
# GenerationRunner.analyse_performance(generation)
# GenerationRunner.plot_starts_and_targets(generation)
# GenerationRunner.plot_target_dates_histogram(generation)

print("Finished")
