from Dataset import BuoyForecastError
from ForecastHindcastDataset import ForecastHindcastDataset
from UNet import UNet
from Generator import Generator
from Discriminator import Discriminator
from utils import sparse_mse, total_variation, mass_conservation

import wandb
import os
import argparse
import datetime
import yaml
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F
from typing import Dict, Callable, Any
from warnings import warn
from tqdm import tqdm
import matplotlib.pyplot as plt


# TODO: overfit on single batch
# TODO: vis fixed batch during course of training
# TODO: verify loss @ init

now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def get_model(model_type: str, model_configs: Dict, device: str) -> Callable:
    if model_type == "unet":
        model = UNet(in_channels=model_configs["in_channels"],
                     out_channels=model_configs["out_channels"],
                     features=model_configs["features"],
                     dropout=model_configs["dropout"])
    elif model_type == "generator":
        model = Generator(in_channels=model_configs["in_channels"],
                          features=model_configs["features"],
                          dropout=model_configs["dropout"])
    return model.to(device)


def get_dataset(dataset_type: str, dataset_configs: Dict) -> Callable:
    if dataset_type == "forecastbuoy":
        dataset = BuoyForecastError(dataset_configs["forecasts"],
                                    dataset_configs["ground_truth"],
                                    dataset_configs["sparse_type"],
                                    dataset_configs["len"])
    elif dataset_type == "forecasthindcast":
        dataset = ForecastHindcastDataset(dataset_configs["forecasts"],
                                          dataset_configs["hindcasts"])
    return dataset


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
        loss = F.mse_loss(predictions, target, reduction="sum")
    elif type == "sparse_mse":
        loss = sparse_mse(predictions, target)
    elif type == "sparse_mse_and_tv":
        loss = sparse_mse(predictions, target) + 0.001*total_variation(predictions)
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
                wandb.log({"train_loss": loss.item()})

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
                wandb.log({"val_loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    print(f"Validation avg loss: {avg_loss:.2f}")
    return avg_loss


def clean_up_training(model, dataloader, base_path: str):
    # report best losses
    # potential plots
    wandb.finish()
    torch.save(model, os.path.join(base_path, now_str))

    # hack to save overfitted sample
    model.eval()
    training_example = next(iter(dataloader))[0]
    output_train = model(training_example).cpu().detach().numpy()
    plt.imsave(os.path.join(base_path, "training_sample.png"), training_example[0, 0])
    plt.imsave(os.path.join(base_path, "training_reconstruction.png"), output_train[0, 0])

    validation_example = next(iter(dataloader))[1]
    output_val = model(validation_example).cpu().detach().numpy()
    plt.imsave(os.path.join(base_path, "validation_sample.png"), validation_example[0, 0])
    plt.imsave(os.path.join(base_path, "validation_reconstruction.png"), output_val[0, 0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="specify the file config for model and training")
    config_file = parser.parse_args().config_file
    all_cfgs = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    wandb_cfgs = {"mode": all_cfgs.get("wandb_mode", "online")}
    wandb.init(project="Generative Models for Realistic Simulation of Ocean Currents",
               entity="ocean-platform-control",
               config=all_cfgs,
               tags="cool stuff",
               **wandb_cfgs)
    wandb.save(config_file)

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
    dataset = get_dataset(all_cfgs["dataset_type"], cfgs_dataset)
    print(f"Using {all_cfgs['dataset_type']} dataset with {cfgs_dataset}.")
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
    torch.onnx.export(model, torch.randn(1, 2, 256, 256), "/home/jonas/Downloads/my_model.onnx")

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
        clean_up_training(model, train_loader, all_cfgs["save_base_path"])


if __name__ == "__main__":
    main()
