import wandb

sweep_configuration = {
    'method': 'bayes',
    'name': 'big_tile_medium_ds',
    'metric': {
        'goal': 'maximize',
        'name': 'validation_r2_validation_merged'
    },
    'parameters': {
        'batch_size': {'values': [16, 32, 64, 128]},
        'epochs': {'values': [20]},
        'lr': {'max': 0.01, 'min': 0.00001},
        'model_error': {'values': [True, False]},
        'ch_sz': {'values': [[6, 12, 24, 48, 48], [6, 18, 36, 48, 64], [6, 12, 12, 24, 48], [6, 32, 64, 128, 256]]},
        'downsizing_method': {'values': ['maxpool', 'conv', 'avgpool']},
        'dropout_encoder': {'distribution': 'uniform', 'max': 0.3, 'min': 0.01},
        'dropout_bottom': {'distribution': 'uniform', 'max': 0.8, 'min': 0.01},
        'dropout_decoder': {'distribution': 'uniform', 'max': 0.8, 'min': 0.01},
        'weight_decay': {'distribution': 'uniform', 'min': 0.00001, 'max': 0.01},
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project='Seaweed_forecast_improvement')
