import pickle
import sys

from ocean_navigation_simulator.ocean_observer.models.OceanCurrentRunner import (
    main,
)

all_res = {}
export_path = "./ablation_study/all_results_testing_set_ablation_study.pkl"
for size_ds in ["small", "medium", "big"]:
    for size_tile in ["medium", "big"]:
        for gp_enabled in ["", "_without_gp"]:
            # Gather config from sweep.
            type_tile_ds = f"{size_tile}_tile_{size_ds}_ds{gp_enabled}"
            sys.argv = (
                "load_and_eval_model.py"
                + " "
                + f"--file-configs=ablation_study/configs_NN/CNN_GP/CNN_ablation_study_{type_tile_ds}"
            ).split()
            all_metrics = main(
                setup_wandb_parameters_sweep=False,
                evaluate_only=True,
                model_to_load=f"ablation_study/CNN_models_saved/{type_tile_ds}.h5",
                json_model_from_wandb=f"ablation_study/CNN_models_saved/{type_tile_ds}.json",
                enable_wandb=False,
                testing_folder="./data_ablation_study/GP_sampled/testing/",
            )
            all_res[type_tile_ds] = all_metrics
with open(export_path, "wb") as f:
    pickle.dump(all_res, f)
