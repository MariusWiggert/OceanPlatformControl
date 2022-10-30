import os

import wandb

os.environ['WANDB_NOTEBOOK_NAME'] = "Seaweed_forecast_improvement"

sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize',
        'name': 'validation_r2_validation_merged'
    },
    'parameters': {
        'batch_size': {'values': [8, 16, 32, 64, 128]},
        'epochs': {'values': [20]},
        'lr': {'max': 0.01, 'min': 0.00001},
        'model_error': {'values': [True, False]},
        'activation': {'values': ['relu', 'elu', 'leakyrelu']},
        'ch_sz': {'values': [[6, 12, 24, 36, 48], [6, 12, 24, 48, 96], [6, 18, 36, 60, 112], [6, 24, 48, 96, 192],
                             [6, 32, 64, 64, 128], [6, 32, 64, 128, 256]]},
        'downsizing_method': {'values': ['maxpool', 'conv', 'avgpool']},
        'dropout_encoder': {'distribution': 'uniform', 'max': 0.9, 'min': 0.01},
        'dropout_bottom': {'distribution': 'uniform', 'max': 0.9, 'min': 0.01},
        'dropout_decoder': {'distribution': 'uniform', 'max': 0.9, 'min': 0.01},
        'weight_decay': {'distribution': 'uniform', 'min': 0.00001, 'max': 0.1},
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project='Seaweed_forecast_improvement')
