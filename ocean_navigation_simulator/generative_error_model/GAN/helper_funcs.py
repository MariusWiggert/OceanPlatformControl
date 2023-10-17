import argparse
import datetime
import os
from typing import Any, Callable, Dict, Optional, Tuple
from warnings import warn

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

import wandb
from ocean_navigation_simulator.generative_error_model.GAN.BuoyForecastDataset import (
    BuoyForecastErrorNpy,
)
from ocean_navigation_simulator.generative_error_model.GAN.Discriminator import (
    Discriminator,
)
from ocean_navigation_simulator.generative_error_model.GAN.ForecastHindcastDataset import (
    ForecastHindcastDatasetNpy,
)
from ocean_navigation_simulator.generative_error_model.GAN.Generator import (
    Generator,
    GeneratorSimplified,
)


def initialize(sweep: bool, test: bool = False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, help="specify the file config for model and training"
    )
    config_file = parser.parse_args().config_file
    all_cfgs = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    wandb_cfgs = {"mode": all_cfgs.get("wandb_mode", "online")}
    if test:
        wandb_cfgs = {"mode": "offline"}
    # add model saving name to cfgs
    all_cfgs["model_save_name"] = f"{get_now_string()}.pth"
    wandb.init(
        project="Generative Models for Realistic Simulation of Ocean Currents",
        entity="ocean-platform-control",
        tags=f"test={test}",
        **wandb_cfgs,
    )
    # update wandb configs
    if sweep:
        all_cfgs |= sweep_set_parameter(all_cfgs)
        wandb.config.update(all_cfgs)
    # if all_cfgs["wandb_mode"] == "online":
    # wandb.run.name = f"{all_cfgs['train']['loss']['gen']}" + \
    #                  f"_{all_cfgs['dataset']['area']}" + \
    #                  f"_{all_cfgs['dataset']['len']}" + \
    #                  f"_{all_cfgs['dataset']['concat_len']}"
    wandb.config.update(all_cfgs)

    # # define metrics for different charts
    # metrics = {"real_acc": "epoch", "fake_acc": "epoch", "train_loss_gen": "epoch", "train_loss_disc": "epoch",
    #            "gen_lr": "epoch", "disc_lr": "epoch", "vector_correlation": "epoch", "rmse": "epoch",
    #            "val_loss": "epoch", "real_acc_batch": "train_steps", "fake_acc_batch": "train_steps",
    #            "fixed_batch_predictions": "epochs", "fixed_batch_gt": "epochs"}
    # for metric, x_axis in metrics.items():
    #     wandb.run.define_metric(metric, step_metric=x_axis)
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
        model = Generator(**model_configs["model_settings"])
    elif model_type == "discriminator":
        model = Discriminator(**model_configs["model_settings"])
    elif model_type == "generator_simplified":
        model = GeneratorSimplified(**model_configs["model_settings"])
    else:
        raise ValueError("Specified model type not available!")
    return model.to(device)


def get_data(dataset_type: str, dataset_configs: Dict, train_configs: Dict) -> Tuple:
    """Convenience function. Selects dataset. Create dataloaders."""

    dataset = _get_dataset(dataset_type, dataset_configs)
    return _get_dataloaders(dataset, dataset_configs, train_configs)


def _get_dataset(dataset_type: str, dataset_configs: Dict) -> Callable:
    """To train the complete model different datasets are used. This function
    handles which dataset to use."""

    if dataset_type == "forecastbuoy":
        dataset = BuoyForecastErrorNpy(
            dataset_configs["forecasts"],
            dataset_configs["ground_truth"],
            dataset_configs["area"],
            concat_len=dataset_configs["concat_len"],
            dataset_names=dataset_configs["dataset_names"],
            transform=dataset_configs["transform"],
        )
    elif dataset_type == "forecasthindcast":
        dataset = ForecastHindcastDatasetNpy(
            dataset_configs["forecasts"],
            dataset_configs["hindcasts"],
            dataset_configs["area"],
            dataset_configs["concat_len"],
        )
    # print(f"Using {dataset_type} dataset with {dataset_configs}.")
    return dataset


def _get_dataloaders(dataset: Dataset, dataset_configs: Dict, train_configs: Dict) -> Tuple:
    """Creates Dataloaders according to yaml config.
    Dataset splits are determined on a per-area basis in order to guarantee an equal number of examples
    in each set from each specified area.
    If dataset len is specified to be smaller than the entire dataset (all areas added), then subsets
    can be either random or non-random."""

    rng = np.random.default_rng(12345)
    area_lens = dataset.area_lens
    splits = np.array(dataset_configs["split"])
    prev_dataset_len = 0
    acc_target_len = 0
    train_idx, val_idx, fixed_batch_idx = [], [], []
    fixed_batch_size = int(4 / len(area_lens.keys()))
    for area, length in area_lens.items():
        # get split sizes for the whole dataset.
        train_size = int(splits[0] * length)
        val_size = length - train_size

        # target split sizes equal if using whole dataset, otherwise target split sizes are calculated.
        if dataset_configs["len"] == "None":
            target_len = int(length / len(area_lens.keys()))
            train_target_size = train_size
            val_target_size = val_size
        else:
            target_len = int(dataset_configs["len"] / len(area_lens.keys()))
            train_target_size = int(splits[0] * target_len)
            val_target_size = target_len - train_target_size

        train_idx.extend(np.array(range(acc_target_len, acc_target_len + train_target_size)))
        remaining_idx = np.array(
            range(
                acc_target_len + train_target_size,
                acc_target_len + train_target_size + val_target_size,
            )
        )

        # pick randomly which samples go in the fixed batch
        temp_idx = rng.choice(np.arange(0, len(remaining_idx)), fixed_batch_size, replace=False)
        val_idx.extend(np.delete(remaining_idx, temp_idx))
        fixed_batch_idx.extend(remaining_idx[temp_idx])
        prev_dataset_len += length
        acc_target_len += target_len

    # create subsets based on calculated idx for each dataset
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    fixed_batch_set = torch.utils.data.Subset(dataset, fixed_batch_idx)

    print(
        f"=> All sets using {len(train_set) + len(val_set) + len(fixed_batch_set)} "
        + f"of {len(dataset)} available samples."
    )
    print(f"=> Data from {dataset_configs['area']}.")

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=train_configs["batch_size"],
        shuffle=dataset_configs["shuffle"],
    )
    val_loader = DataLoader(dataset=val_set, batch_size=train_configs["batch_size"], shuffle=False)
    fixed_batch = DataLoader(
        dataset=fixed_batch_set, batch_size=train_configs["batch_size"], shuffle=False
    )
    return train_loader, val_loader, fixed_batch


def get_test_data(dataset_type: str, dataset_configs: Dict, train_configs: Dict):
    """Get specific part of data for test."""
    dataset = _get_dataset(dataset_type, dataset_configs)
    test_loader = DataLoader(
        dataset=dataset, batch_size=train_configs["test_batch_size"], shuffle=False
    )
    return test_loader


def get_optimizer(model, name: str, lr: float, kwargs_optimizer: Optional[dict[str, Any]] = None):
    """Does what it says on the tin."""

    if kwargs_optimizer is None:
        kwargs_optimizer = dict()
    kwargs_optimizer["lr"] = float(lr)
    # print(f"Optimizer params: {args_optimizer}")
    if name.lower() == "adam":
        return optim.Adam(model.parameters(), **kwargs_optimizer)
    elif name.lower() == "sgd":
        return optim.SGD(model.parameters(), **kwargs_optimizer)
    elif name.lower() == "rmsprop":
        return optim.RMSprop(model.parameters(), **kwargs_optimizer)
    raise warn("No optimizer!")


def get_scheduler(optimizer, scheduler_configs: Dict):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        lr_configs         -- dictionary defining parameters
    """
    if scheduler_configs["scheduler_type"] == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )
    else:
        return NotImplementedError(
            "learning rate policy [%s] is not implemented", scheduler_configs["scheduler_type"]
        )
    return scheduler


def save_input_output_pairs(
    data: torch.tensor, output: torch.tensor, save_root: str, save_dir: str, idx: int
) -> str:
    """Saves input-output pairs for testing."""

    save_dir = save_dir.split(".")[0]
    save_dir = os.path.join(save_root, save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # pad idx
    idx = str(idx + 1).zfill(4)

    save_file_path = os.path.join(save_dir, f"input_{idx}.npy")
    _save_data(data, save_file_path)
    save_file_path = os.path.join(save_dir, f"output_{idx}.npy")
    _save_data(output, save_file_path)

    return save_dir


def _save_data(data: torch.tensor, save_file_path: str) -> None:
    data = data.cpu().detach().numpy()
    np.save(save_file_path, data)


def enable_dropout(m, layers: list):
    """Enable specific Dropout layers in generator like in Pix2pix for validation/testing."""

    for name, layer in m.named_modules():
        if layer.__class__.__name__.startswith("Dropout") and (
            "up" in name and any(str(x) in name for x in layers)
        ):
            layer.train()
    # print("-> Enabled dropout in gen")


def init_decoder_weights(generator):
    """Reset weights in decoder part of generator after loading entire pre-trained generator weights."""

    with torch.no_grad():
        for name, layer in generator.named_modules():
            # print(name, type(layer))
            if ("up" in name and "conv.0" in name) or "final_up.0" in name:
                # print(name, type(layer))
                nn.init.xavier_normal_(layer.weight.data, gain=0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias.data, 0.0)
    print("=> Reset decoder weights")


def freeze_encoder_weights(generator):
    """Freezes encoder part of the generator."""

    with torch.no_grad():
        for name, param in generator.named_parameters():
            if name.find("down") != -1 or name.find("bottleneck") != -1:
                param.requires_grad = False
            else:
                param.requires_grad = True
        print("=> Frozen generator encoder")


def gradient_penalty(critic, real, fake, data=None, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    if data is not None:
        mixed_scores = critic(interpolated_images, data)
    else:
        mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def get_latent_vec(shape):
    return torch.randn(shape)


class SeededMasking:
    def __init__(self, seed: int):
        self.seed = seed
        self.gen = torch.Generator(device="cpu")
        self.rand_gen = self.gen.manual_seed(seed)
        self.prev_mask = None

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rand_gen = self.gen.manual_seed(seed)
            self.seed = seed
        else:
            self.rand_gen = self.gen.manual_seed(self.seed)
        # need to reset size otherwise throws error if last batch_size was not full batch
        self.prev_mask = None

    def get_mask(self, mask_shape: tuple, keep_fraction: float):
        if keep_fraction > 1.0:
            raise ValueError("Please specify fraction in interval [0, 1].")
        mask = torch.zeros(mask_shape).flatten()
        mask[: int(keep_fraction * mask.shape[0])] = 1
        rand_idx = torch.randperm(mask.shape[0], generator=self.rand_gen)
        mask = torch.reshape(mask[rand_idx], mask_shape)
        return mask

    def get_perturbed_mask(
        self,
        perturb_prob: float,
        mask_shape: Optional[tuple] = None,
        keep_fraction: Optional[float] = None,
    ):
        if self.prev_mask is None:
            if mask_shape is None or keep_fraction is None:
                assert ValueError(
                    "Need to provide mask_shape and keep_fraction if SeededMasking.prev_mask = None."
                )
            mask = self.get_mask(mask_shape, keep_fraction)
            self.prev_mask = mask
            return mask
        else:
            mask = self._perturb_mask(perturb_prob, mask_shape)
            return mask

    def _perturb_mask(self, perturb_prob: float, mask_shape: tuple):
        if perturb_prob > 1.0:
            raise ValueError("Please specify fraction in interval [0, 1].")
        # decide whether to perturb or not
        if np.random.rand(1)[0] < perturb_prob:
            # if final batch smaller than batch_size
            self.prev_mask = self.prev_mask[: mask_shape[0]]
            # get indices of 1s of prev mask
            idx = (self.prev_mask == 1).nonzero()
            # perturb indices randomly
            idx[:, 2:] += np.random.randint(-1, 2, size=(idx.shape[2:]))  # note high is exclusive
            idx[:, 2:] %= mask_shape[-1]
            # assign value at new indices a 1
            mask = torch.zeros(mask_shape)
            # convert indices into 1D tensor
            b, c, h, w = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
            helper = (
                b * np.prod(mask.shape[1:])
                + c * np.prod(mask.shape[2:])
                + h * np.prod(mask.shape[-1])
                + w
            )
            # assign 1 to indices
            mask.flatten().scatter_(dim=0, index=helper, value=1).reshape_as(mask)
        else:
            mask = self.prev_mask[: mask_shape[0]]
        self.prev_mask = mask
        return mask
