from Dataset import BuoyForecastError
from UNet import UNet

import wandb
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torch import optim
from typing import Dict, Callable, Any
from warnings import warn


def get_model(model_type: str, model_configs: Dict, device: str) -> Callable:
    if model_type == "unet":
        model = UNet(in_channels=model_configs["in_channels"],
                     out_channels=model_configs["out_channels"],
                     features=model_configs["features"])
    elif model_type == "gan":
        pass
    return model.to(device)


def get_optimizer(model, name: str, args_optimizer: dict[str, Any], lr: float):
    args_optimizer['lr'] = lr
    if name.lower() == "adam":
        return optim.Adam(model.parameters(), **args_optimizer)
    raise warn("No optimizer!")
    return None


def train():
    # # log metrics inside loop.
    # wandb.log({"loss": loss})
    return 1


def validation():
    return 1


def clean_up_training():
    pass


def main():
    # wandb.init(project="Generative Models for Realistic Simulation of Ocean Currents",
    #            entity="ocean-platform-control")
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="specify the file config for model and training")
    config_file = parser.parse_args().file_configs
    # wandb.config = load dict from yaml
    all_cfgs = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}.")

    model_type = all_cfgs["model"]
    cfgs_model = all_cfgs[model_type]
    cfgs_dataset = all_cfgs["dataset"]
    cfgs_train = all_cfgs["train"]

    # load training data
    dataset = BuoyForecastError(cfgs_dataset["forecasts"], cfgs_dataset["ground_truth"])
    dataset_len = dataset.__len__()
    train_set, val_set = torch.utils.data.random_split(dataset,
                                                       [0.8*dataset_len, 0.2*dataset_len],
                                                       generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(dataset=train_set, batch_size=cfgs_train["batch_size"], shuffle=cfgs_dataset["shuffle"])
    val_loader = DataLoader(dataset=val_set, batch_size=cfgs_train["test_batch_size"], shuffle=cfgs_dataset["shuffle"])

    # define model
    print(f"Using: {model_type}.")
    model = get_model(model_type, cfgs_model, device)

    optimizer = get_optimizer(model, "adam", {}, lr=cfgs_train["learning_rate"])

    train_losses, val_losses = list(), list()
    try:
        for epoch in range(1, cfgs_train["epochs"] + 1):
            metrics = {}

            loss = train()
            train_losses.append(loss)
            metrics |= {"train_loss": loss}

            loss = validation()
            val_losses.append(loss)
            metrics |= {"val_loss": loss}

            wandb.log(metrics)

    finally:
        clean_up_training()




if __name__ == "__main__":
    main()


