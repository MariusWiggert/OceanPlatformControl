import json
import os

import wandb

wandb.init(
    project="seaweed-rl",
    entity="jeromejeannin",
    dir="~",
    name="test",
)
folder = "/seaweed-storage/experiments/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/increased_area_2022_10_29_01_20_29/results/"
#
for i, file in enumerate(sorted(os.listdir(folder))):
    print(folder + file)
    with open(
        "/seaweed-storage/experiments/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/increased_area_2022_10_29_01_20_29/results/epoch1.json",
        "rt",
    ) as f:
        result = json.load(f)
    # wandb.log()
