from Generator import Generator
from Discriminator import Discriminator
from BuoyForecastDataset import BuoyForecastErrorNpy
from ForecastHindcastDataset import ForecastHindcastDatasetNpy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Callable, Tuple, Any, Optional
import numpy as np
from warnings import warn
import datetime
import yaml
import wandb
import argparse


def initialize(sweep: bool, test: bool = False, tag: Optional[str] = "train"):
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="specify the file config for model and training")
    config_file = parser.parse_args().config_file
    all_cfgs = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    wandb_cfgs = {"mode": all_cfgs.get("wandb_mode", "online")}
    # add model saving name to cfgs
    all_cfgs["model_save_name"] = f"{get_now_string()}.pth"
    wandb.init(project="Generative Models for Realistic Simulation of Ocean Currents",
               entity="ocean-platform-control",
               tags=f"test={test}",
               **wandb_cfgs)
    # update wandb configs
    if sweep:
        all_cfgs |= sweep_set_parameter(all_cfgs)
        wandb.config.update(all_cfgs)
    wandb.run.name = f"{all_cfgs['train']['loss']['types']}" + \
                     f"_{all_cfgs[all_cfgs['model']]['norm_type']}" + \
                     f"_{all_cfgs['dataset']['area']}" + \
                     f"_{all_cfgs['dataset']['len']}" + \
                     f"_{all_cfgs['dataset']['random_subsets']}" + \
                     f"_{all_cfgs['dataset']['concat_len']}"
    wandb.config.update(all_cfgs)
    return all_cfgs


def sweep_set_parameter(args):
    args["train"]["batch_size"] = wandb.config.batch_size
    args["train"]["epochs"] = wandb.config.epochs
    args["train"]["learning_rate"] = wandb.config.lr
    args[args["model"]]["init_type"] = wandb.config.weight_init
    args[args["model"]]["norm_type"] = wandb.config.norm_type
    args["train"]["loss"]["types"] = wandb.config.loss_settings[0]
    args["train"]["loss"]["weighting"] = wandb.config.loss_settings[1]
    return args


def get_now_string():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def get_model(model_type: str, model_configs: Dict, device: str) -> nn.Module:
    """Handles which model to use which is specified in config file."""

    if model_type == "generator":
        model = Generator(in_channels=model_configs["in_channels"],
                          out_channels=model_configs["out_channels"],
                          features=model_configs["features"],
                          norm=model_configs["norm_type"],
                          dropout=model_configs["dropout"])
    elif model_type == "discriminator":
        model = Discriminator(in_channels=model_configs["in_channels"],
                              features=model_configs["features"],
                              norm=model_configs["norm_type"])
    return model.to(device)


def get_data(dataset_type: str, dataset_configs: Dict, train_configs: Dict, test=False) -> Tuple:
    """Convenience function. Selects dataset. Create dataloaders."""

    dataset = _get_dataset(dataset_type, dataset_configs)
    return _get_dataloaders(dataset, dataset_configs, train_configs, test)


def _get_dataset(dataset_type: str, dataset_configs: Dict) -> Callable:
    """To train the complete model different datasets are used. This function
    handles which dataset to use."""

    if dataset_type == "forecastbuoy":
        dataset = BuoyForecastErrorNpy(dataset_configs["forecasts"],
                                       dataset_configs["ground_truth"],
                                       dataset_configs["sparse_type"],
                                       dataset_configs["concat_len"])
    elif dataset_type == "forecasthindcast":
        dataset = ForecastHindcastDatasetNpy(dataset_configs["forecasts"],
                                             dataset_configs["hindcasts"],
                                             dataset_configs["area"],
                                             dataset_configs["concat_len"])
    # print(f"Using {dataset_type} dataset with {dataset_configs}.")
    return dataset


def _get_dataloaders(dataset: Dataset, dataset_configs: Dict, train_configs: Dict, test=False) -> Tuple:
    """Creates Dataloaders according to yaml config.
    Dataset splits are determined on a per-area basis in order to guarantee an equal number of examples
    in each set from each specified area.
    If dataset len is specified to be smaller than the entire dataset (all areas added), then subsets
    can be either random or non-random."""

    if test:
        dataset_configs["len"] = "None"

    rng = np.random.default_rng(12345)
    area_lens = dataset.area_lens
    splits = np.array(dataset_configs["split"])
    prev_dataset_len = 0
    acc_target_len = 0
    train_idx, val_idx, fixed_batch_idx, test_idx = [], [], [], []
    fixed_batch_size = int(4/len(area_lens.keys()))
    for area, length in area_lens.items():
        # get split sizes for the whole dataset.
        train_size = int(splits[0] * length)
        val_size = int((splits[1]/splits[1:].sum())*(length - train_size))
        test_size = length - train_size - val_size

        # target split sizes equal if using whole dataset, otherwise target split sizes are calculated.
        if dataset_configs["len"] == "None":
            target_len = int(length / len(area_lens.keys()))
            train_target_size = train_size
            val_target_size = val_size
            test_target_size = test_size
        else:
            target_len = int(dataset_configs["len"] / len(area_lens.keys()))
            train_target_size = int(splits[0] * target_len)
            val_target_size = int((splits[1]/splits[1:].sum())*(target_len - train_target_size))
            test_target_size = target_len - train_target_size - val_target_size

        # get indices for samples for each sub-dataset.
        if dataset_configs["random_subsets"]:
            # if random_subsets -> sample from all possible sub-dataset options.
            train_idx.extend(rng.choice(np.array(range(prev_dataset_len, prev_dataset_len + train_size)),
                             size=train_target_size,
                             replace=False))
            val_idx.extend(rng.choice(np.array(range(prev_dataset_len + train_size, prev_dataset_len + train_size+val_size)),
                           size=val_target_size,
                           replace=False))
            remaining_idx = rng.choice(np.array(range(prev_dataset_len + train_size + val_size,
                                                prev_dataset_len + train_size + val_size + test_size)),
                                       size=test_target_size, replace=False)
        else:
            # if non-random choose adjacent blocks of data for sub-datasets.
            train_idx.extend(np.array(range(acc_target_len, acc_target_len + train_target_size)))
            val_idx.extend(np.array(range(acc_target_len + train_target_size,
                                          acc_target_len + train_target_size + val_target_size)))
            remaining_idx = np.array(range(acc_target_len + train_target_size + val_target_size,
                                           acc_target_len + train_target_size + val_target_size + test_target_size))

        # pick randomly which samples go in the fixed batch
        temp_idx = rng.choice(np.arange(0, len(remaining_idx)), fixed_batch_size, replace=False)
        test_idx.extend(np.delete(remaining_idx, temp_idx))
        fixed_batch_idx.extend(remaining_idx[temp_idx])
        prev_dataset_len += length
        acc_target_len += target_len

    # create subsets based on calculated idx for each dataset
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    fixed_batch_set = torch.utils.data.Subset(dataset, fixed_batch_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)

    if test:
        print(f"-> Using {len(fixed_batch_set) + len(test_set)} of {len(dataset)} available samples.")
        print(f"-> Data from {dataset_configs['area']}.")
    else:
        print(f"-> All sets using {len(train_set) + len(val_set) + len(fixed_batch_set) + len(test_set)} " +
              f"of {len(dataset)} available samples.")
        print(f"-> Data from {dataset_configs['area']}.")

    train_loader = DataLoader(dataset=train_set, batch_size=train_configs["batch_size"], shuffle=dataset_configs["shuffle"])
    val_loader = DataLoader(dataset=val_set, batch_size=train_configs["batch_size"], shuffle=False)
    test_loader = DataLoader(dataset=test_set, batch_size=train_configs["test_batch_size"], shuffle=False)
    fixed_batch = DataLoader(dataset=fixed_batch_set, batch_size=train_configs["batch_size"], shuffle=False)
    return train_loader, val_loader, test_loader, fixed_batch


def get_optimizer(model, name: str, args_optimizer: dict[str, Any], lr: float):
    """Does what it says on the tin."""
    args_optimizer['lr'] = float(lr)
    # print(f"Optimizer params: {args_optimizer}")
    if name.lower() == "adam":
        return optim.Adam(model.parameters(), **args_optimizer)
    raise warn("No optimizer!")


def get_scheduler(optimizer, scheduler_configs: Dict):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        lr_configs         -- dictionary defining parameters
    """
    if scheduler_configs["scheduler_type"] == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', scheduler_configs["scheduler_type"])
    return scheduler