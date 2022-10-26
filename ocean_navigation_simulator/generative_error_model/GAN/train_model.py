from BuoyForecastError import BuoyForecastError
from ForecastHindcastDataset import ForecastHindcastDataset, ForecastHindcastDatasetNpy
from UNet import UNet
from Generator import Generator
from Discriminator import Discriminator
from utils import l1, mse, sparse_mse, total_variation, mass_conservation, \
    init_weights, save_checkpoint, load_checkpoint
from ocean_navigation_simulator.generative_error_model.generative_model_metrics import rmse, vector_correlation

import wandb
import os
import argparse
import datetime
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torch.optim import lr_scheduler
from torchvision.utils import make_grid
from typing import Dict, Callable, Any, Tuple, List, Optional
from warnings import warn
from tqdm import tqdm
import numpy as np


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
                          out_channels=model_configs["out_channels"],
                          features=model_configs["features"],
                          norm=model_configs["norm_type"],
                          dropout=model_configs["dropout"])
    return model.to(device)


def get_data(dataset_type: str, dataset_configs: Dict, train_configs: Dict, test=False) -> Tuple:
    """Convenience function. Selects dataset. Create dataloaders."""

    dataset = _get_dataset(dataset_type, dataset_configs)
    return _get_dataloaders(dataset, dataset_configs, train_configs, test)


def _get_dataset(dataset_type: str, dataset_configs: Dict) -> Callable:
    """To train the complete model different datasets are used. This function
    handles which dataset to use."""

    if dataset_type == "forecastbuoy":
        dataset = BuoyForecastError(dataset_configs["forecasts"],
                                    dataset_configs["ground_truth"],
                                    dataset_configs["sparse_type"],
                                    dataset_configs["len"])
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
        dataset_configs["len"] = None

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
    args_optimizer['lr'] = float(lr)
    # print(f"Optimizer params: {args_optimizer}")
    if name.lower() == "adam":
        return optim.Adam(model.parameters(), **args_optimizer)
    raise warn("No optimizer!")
    return None


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


def predict_fixed_batch(model, dataloader, device):
    """Performs prediction on a fixed batch to be able to assess performance qualitatively.
    This batch is visualized and saved on weights and biases."""

    model.eval()
    with torch.no_grad():
        data = next(iter(dataloader))
        samples = data[0]
        ground_truth = data[1]
        samples = samples.to(device)
        model_output = model(samples).cpu().detach()
        samples = samples.cpu().detach()[:, :2]

    # logging final images to weights and biases
    ground_truth = make_grid(ground_truth, 2)
    predictions = make_grid(model_output, 2)
    samples = make_grid(samples, 2)
    ground_truth = wandb.Image(ground_truth, caption="Fixed batch samples")
    predictions = wandb.Image(predictions, caption="Fixed batch predictions")
    samples = wandb.Image(samples, caption="Fixed batch samples")
    wandb.log({"fixed_batch_gt": ground_truth, "fixed_batch_predictions": predictions, "fixed_batch_inputs": samples})
    model.train()


def get_metrics(metric_names, ground_truth: torch.Tensor, predictions: torch.Tensor) -> Optional[Dict[str, float]]:
    """Computes all specified metrics over the validation set predictions.
    Computes metrics for each pair, sums the result of all pairs in batch and returns metrics
    normalized by batch size."""

    # convert to numpy
    ground_truth = ground_truth.cpu().detach().numpy()
    predictions = predictions.cpu().detach().numpy()

    # compute metrics
    metrics = {metric_name: 0 for metric_name in metric_names}
    for idx in range(ground_truth.shape[0]):
        if "rmse" in metric_names:
            rmse_val = rmse(ground_truth[idx, :2, :, :].squeeze(),
                            predictions[idx, :2, :, :].squeeze())
            metrics["rmse"] += rmse_val
        if "vector_correlation" in metric_names:
            vec_corr_val = vector_correlation(ground_truth[idx, 0, :, :].squeeze(),
                                              ground_truth[idx, 1, :, :].squeeze(),
                                              predictions[idx, 0, :, :].squeeze(),
                                              predictions[idx, 1, :, :].squeeze())
            metrics["vector_correlation"] += vec_corr_val
        else:
            raise NotImplementedError("This specified metric does not exist!")
    metrics = {metric_name: metric_value / ground_truth.shape[0] for metric_name, metric_value in metrics.items()}
    return metrics


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
                train_loss = loss_function(output, target, cfgs_train["loss"]["types"], cfgs_train["loss"]["weighting"])
                total_loss += train_loss.item()

                # perform optim step
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=str(round(train_loss.item() / data.shape[0], 3)))
                # wandb.log({"train_loss": round(train_loss.item() / data.shape[0], 3)})

    avg_loss = total_loss / ((len(dataloader)-1)*cfgs_train["batch_size"] + data.shape[0])
    # print(f"Training avg loss: {avg_loss:.2f}.")
    return avg_loss


def validation(model, dataloader, device, cfgs_train, metrics_names):
    total_loss = 0
    model.eval()
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Validation epoch [{cfgs_train['epoch']}/{cfgs_train['epochs']}]")
            metrics = {metric: 0 for metric in metrics_names}
            metrics_ratio = {metric: 0 for metric in metrics_names}
            for data, target in tepoch:
                data, target = data.to(device), target.to(device)

                output = model(data)
                val_loss = loss_function(output, target, cfgs_train["loss"]["types"], cfgs_train["loss"]["weighting"])
                total_loss += val_loss.item()

                # get metrics and ratio of metrics for generated outputs
                metric_values = get_metrics(metrics_names, target, output)
                metric_values_baseline = get_metrics(metrics_names, target, data)
                if metric_values_baseline["rmse"] == 0:
                    print("RMSE of baseline is 0!")
                if metric_values_baseline["vector_correlation"] == 2:
                    print("Vector correlation of baseline is 2!")
                for metric_name in metrics_names:
                    metrics[metric_name] += metric_values[metric_name]
                    metrics_ratio[metric_name] += metric_values[metric_name] / (metric_values_baseline[metric_name]+1e-8)

                tepoch.set_postfix(loss=str(round(val_loss.item() / data.shape[0], 3)))
                # wandb.log({"val_loss": round(val_loss.item() / data.shape[0], 3)})

    metrics = {metric_name: metric_value/len(dataloader) for metric_name, metric_value in metrics.items()}
    metrics_ratio = {metric_name + "_ratio": metric_value/len(dataloader) for metric_name, metric_value in metrics_ratio.items()}
    wandb.log(metrics)
    wandb.log(metrics_ratio)
    avg_loss = total_loss / ((len(dataloader)-1)*cfgs_train["batch_size"] + data.shape[0])
    # print(f"Validation avg loss: {avg_loss:.2f}")
    return avg_loss


def clean_up_training(model, optimizer, dataloader, base_path: str, device: str):
    """Saves final model. Saves plot for fixed batch."""

    save_checkpoint(model, optimizer, f"{os.path.join(base_path, now_str)}.pth")
    predict_fixed_batch(model, dataloader, device)
    wandb.finish()


def initialize(test: bool = False):
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="specify the file config for model and training")
    config_file = parser.parse_args().config_file
    all_cfgs = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    wandb_cfgs = {"mode": all_cfgs.get("wandb_mode", "online")}
    # add model saving name to cfgs
    all_cfgs["model_save_name"] = f"{now_str}.pth"
    wandb.init(project="Generative Models for Realistic Simulation of Ocean Currents",
               entity="ocean-platform-control",
               name=f"{all_cfgs['train']['loss']['types']}" +
                    f"_{all_cfgs[all_cfgs['model']]['norm_type']}" +
                    f"_{all_cfgs['dataset']['area']}" +
                    f"_{all_cfgs['dataset']['len']}" +
                    f"_{all_cfgs['dataset']['random_subsets']}" +
                    f"_{all_cfgs['dataset']['concat_len']}",
               tags=f"test={test}",
               **wandb_cfgs)
    wandb.save(config_file)
    return all_cfgs


def sweep_set_parameter(args):
    args["train"]["batch_size"] = wandb.config.batch_size
    args["train"]["epochs"] = wandb.config.epochs
    args["train"]["learning_rate"] = wandb.config.lr
    args[args["model"]]["init_type"] = wandb.config.weight_init
    args[args["model"]]["norm_type"] = wandb.config.norm_type
    args["train"]["loss"]["types"] = wandb.config.loss_type
    args["train"]["loss"]["weighting"] = wandb.config.loss_weighting
    return args


def main(sweep: Optional[bool] = False):
    all_cfgs = initialize()
    print("####### Start Training #######")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> Running on: {device}.")
    all_cfgs["device"] = device

    # seed for reproducibility
    torch.manual_seed(0)

    # update wandb configs
    if sweep:
        all_cfgs = sweep_set_parameter(all_cfgs)
        wandb.config.update(all_cfgs)

    # simplify config access
    model_type = all_cfgs["model"]
    cfgs_model = all_cfgs[model_type]
    cfgs_dataset = all_cfgs["dataset"]
    cfgs_train = all_cfgs["train"]
    cfgs_optimizer = all_cfgs["optimizer"]
    cfgs_lr_scheduler = all_cfgs["train"]["lr_scheduler_configs"]

    # load training data
    train_loader, val_loader, _, fixed_batch_loader = get_data(all_cfgs["dataset_type"], cfgs_dataset, cfgs_train)

    # define model and optimizer and load from checkpoint if specified
    print(f"-> Model: {model_type}.")
    print(f"-> Losses: {cfgs_train['loss']['types']} with weightings {cfgs_train['loss']['weighting']}.")
    model = get_model(model_type, cfgs_model, device)
    optimizer = get_optimizer(model, cfgs_optimizer["name"], cfgs_optimizer["parameters"], lr=cfgs_train["learning_rate"])
    if cfgs_lr_scheduler["value"]:
        lr_scheduler = get_scheduler(optimizer, cfgs_lr_scheduler)
    if all_cfgs["load_from_chkpt"]["value"]:
        checkpoint_path = os.path.join(all_cfgs["save_base_path"], all_cfgs["load_from_chkpt"]["file_name"])
        load_checkpoint(checkpoint_path, model, optimizer, cfgs_train["learning_rate"], device)
    else:
        init_weights(model, init_type=cfgs_model["init_type"], init_gain=cfgs_model["init_gain"])
    # torch.onnx.export(model, torch.randn(1, 2, 256, 256), "/home/jonas/Downloads/my_model.onnx")

    train_losses, val_losses, lrs = list(), list(), list
    try:
        for epoch in range(1, cfgs_train["epochs"] + 1):
            print()
            wandb.log({"lr": optimizer.param_groups[0]["lr"]})
            metrics = {}
            cfgs_train["epoch"] = epoch

            train_loss = train(model, train_loader, device, optimizer, cfgs_train)
            train_losses.append(train_loss)
            metrics |= {"train_loss": train_loss}
            wandb.log({"train_loss": train_loss})

            if len(val_loader) != 0:
                val_loss = validation(model, val_loader, device, cfgs_train, all_cfgs["metrics"])
                val_losses.append(val_loss)
                metrics |= {"val_loss": val_loss}
                wandb.log({"val_loss": val_loss})
            if cfgs_lr_scheduler["value"]:
                lr_scheduler.step(val_loss)

            wandb.log(metrics)
            # print(f"Epoch metrics: {metrics}.")

            if epoch % 5 == 0 or epoch == 1:
                predict_fixed_batch(model, fixed_batch_loader, device)

    finally:
        clean_up_training(model, optimizer, fixed_batch_loader, all_cfgs["save_base_path"], device)


def test():
    all_cfgs = initialize(test=True)
    print("####### Start Testing #######")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # simplify config access
    cfgs_model = all_cfgs[all_cfgs["model"]]
    cfgs_dataset = all_cfgs["dataset"]
    cfgs_train = all_cfgs["train"]

    # load training data
    _, _, test_loader, _ = get_data(all_cfgs["dataset_type"], cfgs_dataset, cfgs_train, test=True)

    # load model
    model = get_model(all_cfgs["model"], cfgs_model, device)
    checkpoint_path = os.path.join(all_cfgs["save_base_path"], all_cfgs["load_from_chkpt"]["file_name"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    print()
    cfgs_train["epoch"] = 1
    _ = validation(model, test_loader, device, cfgs_train, all_cfgs["metrics"])


if __name__ == "__main__":
    main()
    # test()
