import wandb
from train_model import main


sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric':
    {
        'goal': 'minimize',
        'name': 'rmse'
    },
    'parameters':
    {
        'batch_size': {'values': [16, 32, 64, 128]},
        'epochs': {'values': [30]},
        'lr': {'max': 0.1, 'min': 0.0001},
        'weight_init': {'values': ['xavier', 'normal']},
        'norm_type': {'values': ['instance', 'batch']},
        'loss_type': {'values': [['mse'], ['l1']]},
        'loss_weighting': {'values': [[1], [1]]}
     }
}


def sweep():
    # Initialize sweep by passing in config. (Optional) Provide a name of the project.
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='cool-sweepedy-sweeeeep')
    # Start sweep job.
    config_file = "scenarios/generative_error_model/config_dl_training.yaml"
    wandb.agent(sweep_id, lambda: main(sweep=True), count=4)


if __name__ == "__main__":
    sweep()
