import wandb
from train_gan import main


sweep_configuration = {
    "method": "bayes",
    "name": "cool-sweepedy-sweeeep",
    "metric": {"goal": "minimize", "name": "rmse"},
    "parameters": {
        "batch_size": {"values": [16, 32, 64, 128]},
        "epochs": {"values": [30]},
        "lr": {"max": 0.01, "min": 0.0001},
        "weight_init": {"values": ["xavier", "normal"]},
        "norm_type": {"values": ["instance", "batch", "no_norm"]},
        "loss_settings": {"values": [(["mse"], [1]), (["l1"], [1])]},
    },
}


def sweep(sweep_id: str = None):
    if sweep_id is None:
        # Initialize sweep by passing in config. (Optional) Provide a name of the project.
        sweep_id = wandb.sweep(
            sweep=sweep_configuration,
            project="Generative Models for Realistic Simulation of Ocean Currents",
        )

    # Start sweep job.
    wandb.agent(sweep_id, lambda: main(sweep=True), count=10)


if __name__ == "__main__":
    sweep()
