from BuoyForecastError import BuoyForecastError
from ForecastHindcastDataset import ForecastHindcastDataset
from UNet import UNet
from Generator import Generator
from Discriminator import Discriminator
from utils import l1, mse, sparse_mse, total_variation, mass_conservation, \
    init_weights, save_checkpoint, load_checkpoint

import wandb
import os
import argparse
import datetime
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torchvision.utils import make_grid
from typing import Dict, Callable, Any, Tuple, List
from warnings import warn
from tqdm import tqdm


# TODO: vis fixed batch during course of training
# TODO: weight init -> pix2pix: mean=0 and std=0.02
# TODO: verify loss @ init

now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def get_model(model_type: str, model_configs: Dict, device: str) -> Callable:
    """Handles which model to use which is specified in config file."""

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


def get_data(dataset_type: str, dataset_configs: Dict, train_configs: Dict) -> Tuple:
    """Convenience function. Selects dataset. Create dataloaders."""

    dataset = _get_dataset(dataset_type, dataset_configs)
    return _get_dataloaders(dataset, dataset_configs, train_configs)


def _get_dataset(dataset_type: str, dataset_configs: Dict) -> Callable:
    """To train the complete model different datasets are used. This function
    handles which dataset to use."""

    if dataset_type == "forecastbuoy":
        dataset = BuoyForecastError(dataset_configs["forecasts"],
                                    dataset_configs["ground_truth"],
                                    dataset_configs["sparse_type"],
                                    dataset_configs["len"])
    elif dataset_type == "forecasthindcast":
        dataset = ForecastHindcastDataset(dataset_configs["forecasts"],
                                          dataset_configs["hindcasts"])
    print(f"Using {dataset_type} dataset with {dataset_configs}.")
    return dataset


def _get_dataloaders(dataset: Dataset, dataset_configs: Dict, train_configs: Dict) -> Tuple:
    """Creates Dataloaders according to yaml config."""

    if dataset_configs["len"] < len(dataset):
        dataset_len = int(dataset_configs["len"])
    else:
        dataset_len = len(dataset)
    print(f"Using {dataset_len} of {len(dataset)} available samples.")
    train_size = round(0.6 * dataset_len)
    val_size = int(0.5*(dataset_len - train_size))
    fixed_batch_size = 4 if dataset_len - train_size - val_size > 4 else 1
    test_size = dataset_len - train_size - val_size - fixed_batch_size

    # if using subset of dataset get idx
    dataset_idx = list(range(0, dataset_len))
    dataset = torch.utils.data.Subset(dataset, dataset_idx)
    train_set, val_set, test_set, fixed_batch = torch.utils.data.random_split(dataset,
                                                                              [train_size, val_size, test_size, fixed_batch_size],
                                                                              generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(dataset=train_set, batch_size=train_configs["batch_size"], shuffle=dataset_configs["shuffle"])
    val_loader = DataLoader(dataset=val_set, batch_size=train_configs["batch_size"], shuffle=dataset_configs["shuffle"])
    test_loader = DataLoader(dataset=test_set, batch_size=train_configs["test_batch_size"], shuffle=dataset_configs["shuffle"])
    fixed_batch = DataLoader(dataset=fixed_batch, batch_size=fixed_batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, fixed_batch


def get_optimizer(model, name: str, args_optimizer: dict[str, Any], lr: float):
    args_optimizer['lr'] = float(lr)
    print(f"Optimizer params: {args_optimizer}")
    if name.lower() == "adam":
        return optim.Adam(model.parameters(), **args_optimizer)
    raise warn("No optimizer!")
    return None


def get_scheduler(optimizer, opt_dict: Dict):
    pass


def predict_fixed_batch(model, dataloader, device):
    """Performs prediction on a fixed batch to be able to assess performance qualitatively.
    This batch is visualized and saved on weights and biases."""

    model.eval()
    samples = []
    outputs = []
    for idx, (sample, _) in enumerate(dataloader):
        sample = sample.to(device)
        model_output = model(sample).cpu().detach()
        samples.append(sample[0])
        outputs.append(model_output[0])

    # logging final images to weights and biases
    samples = make_grid(samples, 2)
    outputs = make_grid(outputs, 2)
    images = wandb.Image(samples, caption="Fixed batch samples")
    predictions = wandb.Image(outputs, caption="Fixed batch predictions")
    wandb.log({"fixed_batch_samples": images, "fixed_batch_predictions": predictions})
    model.train()


def loss_function(predictions, target, losses: List[str], loss_weightings: List[float]):
    """Handles which loss is to be used.
    """
    loss = 0
    loss_map = {
        "mse": mse,
        "l1": l1,
        "sparse_mse": sparse_mse,
        "total_variation": total_variation,
        "mass_conservation": mass_conservation
    }

    if len(losses) != len(loss_weightings):
        raise ValueError("'losses' and 'loss_weightings' are not the same length!")

    for loss_type, weight in zip(losses, loss_weightings):
        if loss_type in ["mse", "l1", "sparse_mse"]:
            loss += weight * loss_map[loss_type](predictions, target)
        else:
            loss += weight * loss_map[loss_type](predictions)

    if loss == 0:
        raise warn("Loss is zero!")
    return loss


def train(model: nn.Module, dataloader, device, optimizer, cfgs_train):
    total_loss = 0
    model.train()
    with torch.enable_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Training epoch [{cfgs_train['epoch']}/{cfgs_train['epochs']}]")
            for data, target in tepoch:
                data, target = data.to(device), target.to(device)

                output = model(data)

                # compute loss
                loss = loss_function(output, target, cfgs_train["loss"]["types"], cfgs_train["loss"]["weighting"])
                total_loss += loss.item()

                # perform optim step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=str(round(loss.item(), 3)))
                wandb.log({"train_loss": loss.item()})

    avg_loss = total_loss / len(dataloader)*cfgs_train["batch_size"]
    print(f"Training avg loss: {avg_loss:.2f}.")

    return avg_loss


def validation(model, dataloader, device, cfgs_train):
    total_loss = 0
    model.eval()
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Validation epoch [{cfgs_train['epoch']}/{cfgs_train['epochs']}]")
            for data, target in tepoch:
                data, target = data.to(device), target.to(device)

                output = model(data)
                loss = loss_function(output, target, cfgs_train["loss"]["types"], cfgs_train["loss"]["weighting"])
                total_loss += loss.item()

                tepoch.set_postfix(loss=str(round(loss.item(), 3)))
                wandb.log({"val_loss": loss.item()})

    avg_loss = total_loss / len(dataloader)*cfgs_train["batch_size"]
    print(f"Validation avg loss: {avg_loss:.2f}")
    return avg_loss


def clean_up_training(model, optimizer, dataloader, base_path: str, device: str):
    """Saves final model. Saves plot for fixed batch."""

    save_checkpoint(model, optimizer, f"{os.path.join(base_path, now_str)}.pth")
    predict_fixed_batch(model, dataloader, device)
    wandb.finish()


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
    all_cfgs["device"] = device

    # seed for reproducibility
    torch.manual_seed(0)

    # simply config access
    model_type = all_cfgs["model"]
    cfgs_model = all_cfgs[model_type]
    cfgs_dataset = all_cfgs["dataset"]
    cfgs_train = all_cfgs["train"]
    cfgs_optimizer = all_cfgs["optimizer"]

    # load training data
    train_loader, val_loader, _, fixed_batch_loader = get_data(all_cfgs["dataset_type"], cfgs_dataset, cfgs_train)

    # define model and optimizer and load from checkpoint if specified
    print(f"Using: {model_type}.")
    print(f"Using: {cfgs_train['loss']['types']} with weightings {cfgs_train['loss']['weighting']}.")
    model = get_model(model_type, cfgs_model, device)
    optimizer = get_optimizer(model, cfgs_optimizer["name"], cfgs_optimizer["parameters"], lr=cfgs_train["learning_rate"])
    if all_cfgs["load_from_chkpt"]["value"]:
        checkpoint_path = os.path.join(all_cfgs["save_base_path"], all_cfgs["load_from_chkpt"]["file_name"])
        load_checkpoint(checkpoint_path, model, optimizer, cfgs_train["learning_rate"], device)
    else:
        init_weights(model, init_type=cfgs_model["init_type"], init_gain=cfgs_model["init_gain"])
    # torch.onnx.export(model, torch.randn(1, 2, 256, 256), "/home/jonas/Downloads/my_model.onnx")
    train_losses, val_losses = list(), list()
    try:
        for epoch in range(1, cfgs_train["epochs"] + 1):
            metrics = {}
            cfgs_train["epoch"] = epoch

            loss = train(model, train_loader, device, optimizer, cfgs_train)
            train_losses.append(loss)
            metrics |= {"train_loss": loss}

            if len(val_loader) != 0:
                loss = validation(model, val_loader, device, cfgs_train)
                val_losses.append(loss)
                metrics |= {"val_loss": loss}

            wandb.log(metrics)
            print(f"Epoch metrics: {metrics}.")

            if epoch % 1 == 0:
                predict_fixed_batch(model, fixed_batch_loader, device)

    finally:
        clean_up_training(model, optimizer, fixed_batch_loader, all_cfgs["save_base_path"], device)


if __name__ == "__main__":
    main()
