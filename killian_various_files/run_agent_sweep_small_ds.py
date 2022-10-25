import wandb

from ocean_navigation_simulator.ocean_observer.models.OceanCurrentRunner import main

wandb.agent('hubvlu60', project='Seaweed_forecast_improvement',
            function=lambda: main(setup_wandb_parameters_sweep=True))
