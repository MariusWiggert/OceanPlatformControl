import wandb

sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize',
        'name': 'validation_r2_validation_merged'
    },
    'parameters': {
        'batch_size': {'values': [16, 32, 64, 128]},
        'epochs': {'values': [24]},
        'lr': {'max': 0.01, 'min': 0.00001},
        'model_error': {'values': [True, False]},
        'ch_sz': {'values': [[2, 6, 12, 24, 48], [2, 8, 24, 48, 48], [2, 8, 20, 40, 80], [2, 8, 32, 126, 126]]},
        'downsizing_method': {'values': ['maxpool', 'conv', 'avgpool']},
        'activation': {'values': ['relu', 'elu', 'leakyrelu']},
        'dropout_encoder': {'distribution': 'uniform', 'max': 0.3, 'min': 0.01},
        'dropout_bottom': {'distribution': 'uniform', 'max': 0.8, 'min': 0.01},
        'dropout_decoder': {'distribution': 'uniform', 'max': 0.8, 'min': 0.01},
        'weight_decay': {'distribution': 'uniform', 'min': 0.00001, 'max': 0.01},
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project='Seaweed_forecast_improvement')
