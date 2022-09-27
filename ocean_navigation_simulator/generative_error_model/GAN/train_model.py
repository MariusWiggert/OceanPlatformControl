from Dataset import BuoyForecastError
from UNet import UNet

import wandb
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F
from typing import Dict, Callable, Any
from warnings import warn
from tqdm import tqdm


# TODO: overfit on single batch
# TODO: vis fixed batch during course of training
# TODO: verify loss @ init


def get_model(model_type: str, model_configs: Dict, device: str) -> Callable:
    if model_type == "unet":
        model = UNet(in_channels=model_configs["in_channels"],
                     out_channels=model_configs["out_channels"],
                     features=model_configs["features"])
    elif model_type == "gan":
        pass
    return model.to(device)


def get_optimizer(model, name: str, args_optimizer: dict[str, Any], lr: float):
    args_optimizer['lr'] = float(lr)
    print(f"Optimizer params: {args_optimizer}")
    if name.lower() == "adam":
        return optim.Adam(model.parameters(), **args_optimizer)
    raise warn("No optimizer!")
    return None


def predict_fixed_batch():
    # use Dataset directly to always access the correct data.
    return


def loss_function(predictions, target, type: str = "mse"):
    """Handles which loss is to be used.
    """
    if type == "mse":
        loss = F.mse_loss(predictions, target, reduction="mean")
    elif type == "sparse_mse":
        loss = torch.where(target != 0, (target**2 + predictions**2)/2, 0).sum()
    return loss


def train(model, dataloader, device, optimizer, cfgs_train):
    total_loss = 0
    model.train()
    with torch.enable_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            for data, target in tepoch:
                data, target = data.to(device), target.to(device)

                output = model(data)

                # compute loss
                loss = loss_function(output, target, cfgs_train["loss_type"])
                total_loss += loss.item()

                # perform optim step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=str(round(loss.item(), 3)))
                wandb.log({"loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    print(f"Training avg loss: {avg_loss:.2f}.")

    return avg_loss


def validation(model, dataloader, device, cfgs_train):
    total_loss = 0
    model.eval()
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            for data, target in tepoch:
                data, target = data.to(device), target.to(device)

                output = model(data)
                loss = loss_function(output, target, cfgs_train["loss_type"])
                total_loss += loss.item()

                tepoch.set_postfix(loss=str(round(loss.item(), 3)))

    avg_loss = total_loss / len(dataloader)
    print(f"Validation avg loss: {avg_loss:.2f}")
    return avg_loss


def clean_up_training():
    # report best losses
    # potential plots
    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="specify the file config for model and training")
    config_file = parser.parse_args().config_file
    all_cfgs = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    wandb_cfgs = {"mode": all_cfgs.get("wandb_mode", "online")}
    wandb.init(project="Generative Models for Realistic Simulation of Ocean Currents",
               entity="ocean-platform-control",
               **wandb_cfgs)
    wandb.config = all_cfgs
    print(all_cfgs)
    wandb.save(config_file, base_path="./")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}.")

    # seed for reproducability
    torch.manual_seed(0)

    # simply config access
    model_type = all_cfgs["model"]
    cfgs_model = all_cfgs[model_type]
    cfgs_dataset = all_cfgs["dataset"]
    cfgs_train = all_cfgs["train"]
    cfgs_optimizer = all_cfgs["optimizer"]

    # load training data
    dataset = BuoyForecastError(cfgs_dataset["forecasts"], cfgs_dataset["ground_truth"], cfgs_dataset["len"])
    print(f"Loading forecasts from {cfgs_dataset['forecasts']} and ground truth from {cfgs_dataset['ground_truth']}.")
    dataset_len = len(dataset)
    print(f"Dataset length: {dataset_len}.")
    train_set, val_set = torch.utils.data.random_split(dataset,
                                                       [dataset_len-int(0.2*dataset_len), int(0.2*dataset_len)],
                                                       generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(dataset=train_set, batch_size=cfgs_train["batch_size"], shuffle=cfgs_dataset["shuffle"])
    val_loader = DataLoader(dataset=val_set, batch_size=cfgs_train["test_batch_size"], shuffle=cfgs_dataset["shuffle"])

    # define model
    print(f"Using: {model_type}.")
    print(f"Using: {cfgs_train['loss_type']}.")
    model = get_model(model_type, cfgs_model, device)

    optimizer = get_optimizer(model, cfgs_optimizer["name"], cfgs_optimizer["parameters"], lr=cfgs_train["learning_rate"])

    train_losses, val_losses = list(), list()
    try:
        for epoch in range(1, cfgs_train["epochs"] + 1):
            metrics = {}

            loss = train(model, train_loader, device, optimizer, cfgs_train)
            train_losses.append(loss)
            metrics |= {"train_loss": loss}

            if len(val_loader) != 0:
                loss = validation(model, val_loader, device, cfgs_train)
                val_losses.append(loss)
                metrics |= {"val_loss": loss}

            wandb.log(metrics)
            print(f"Epoch metrics: {metrics}.")

    finally:
        clean_up_training()


if __name__ == "__main__":
    main()
