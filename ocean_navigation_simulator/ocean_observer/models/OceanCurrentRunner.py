from __future__ import print_function

import argparse
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Tuple, List, Any, Optional
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm

from ocean_navigation_simulator.ocean_observer.Other.DotDict import DotDict
from ocean_navigation_simulator.ocean_observer.models.CustomOceanCurrentsFromFiles import (
    CustomOceanCurrentsFromFiles,
)
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentCNN import (
    OceanCurrentCNNSubgrid,
)
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentConvLSTM import (
    OceanCurrentConvLSTM,
)
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentUnetLSTM import (
    OceanCurrentUnetLSTM,
)
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentsMLP import (
    OceanCurrentMLP,
)
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentsRNN import (
    OceanCurrentRNN,
)


# Class used to train the Neural network models

# Custom Collate funtion to remove the NaNs
def collate_fn(batch):
    batch_filtered = list(filter(lambda x: x[0] is not None, batch))
    if not len(batch_filtered):
        return None, None
    return torch.utils.data.dataloader.default_collate(batch_filtered)


# Example of sweep configuration:
sweep_configuration = {
    "method": "bayes",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "validation_r2_validation_merged"},
    "parameters": {
        "batch_size": {"values": [16, 32, 64, 128]},
        "epochs": {"values": [35]},
        "lr": {"max": 0.01, "min": 0.00001},
        "model_error": {"values": [True, False]},
        "ch_sz": {
            "values": [
                [6, 18, 36, 60, 112],
                [6, 24, 48, 96, 192],
                [6, 32, 64, 64, 128],
                [6, 32, 64, 128, 256],
            ]
        },
        "downsizing_method": {"values": ["maxpool", "conv", "avgpool"]},
        "dropout_encoder": {"distribution": "uniform", "max": 0.3, "min": 0.01},
        "dropout_bottom": {"distribution": "uniform", "max": 0.8, "min": 0.01},
        "dropout_decoder": {"distribution": "uniform", "max": 0.8, "min": 0.01},
        "weight_decay": {"distribution": "uniform", "min": 0.00001, "max": 0.01},
    },
}


def compute_conservation_mass_loss(pred, get_all_cells=False):
    # prediction shape: batch_size x channels[u,v] x time x lat [v] = rows x lon [u] = cols

    # top left removes last col, last row
    # top right removes first col, last row
    # bottom left removes last col, first row
    # bottom right removes first col, first row
    # top_left = torch.clone(prediction[..., :-1, :-1])
    # top_right = torch.clone(prediction[..., :-1, 1:])
    # bottom_left = torch.clone(prediction[..., 1:, :-1])
    # bottom_right = torch.clone(prediction[..., 1:, 1:])

    top_left = pred[..., :-1, :-1]
    top_right = pred[..., :-1, 1:]
    bottom_left = pred[..., 1:, :-1]
    bottom_right = pred[..., 1:, 1:]

    all_losses = -top_left[:, [1]] + top_left[:, [0]] - top_right
    all_losses += bottom_left - bottom_right[:, [0]] + bottom_right[:, [1]]
    all_losses = all_losses.sum(axis=1)

    # set nans to 0
    total_size = all_losses.nelement()
    num_nans = torch.isnan(all_losses).sum().item()
    nans = torch.isnan(all_losses)
    all_losses[nans] = 0
    res = 0.5 * (
        F.mse_loss(all_losses, torch.zeros_like(all_losses), reduction="sum")
        / (total_size - num_nans)
    )
    if get_all_cells:
        all_losses[nans] = math.nan
        return res, all_losses
    return res


def compute_burgers_loss(prediction, Re=math.pi / 0.01, get_all_cells=True):
    # Add the boundaries
    X, Y = prediction.shape[-2:]
    batches = len(prediction)
    boundary_u_t0 = torch.tensor(
        [
            [
                [math.sin(math.pi * x / X) * math.cos(math.pi * y / Y) for y in range(Y)]
                for x in range(X)
            ]
        ]
    ).expand(batches, -1, -1)
    boundary_v_t0 = torch.tensor(
        [
            [
                [math.cos(math.pi * x / X) * math.sin(math.pi * y / Y) for y in range(Y)]
                for x in range(X)
            ]
        ]
    ).expand(batches, -1, -1)
    # only pad t in 1 dim, pad lon,lat in two dimensions
    prediction_padded = F.pad(prediction, (1, 1, 1, 1, 1, 0, 0, 0, 0, 0))

    # apply the boundary counditions on the time side
    prediction_padded[:, 0, 0, 1:-1, 1:-1] = boundary_u_t0
    prediction_padded[:, 1, 0, 1:-1, 1:-1] = boundary_v_t0

    u, v = prediction[:, [0]], prediction[:, [1]]
    dt = (prediction_padded[:, :, 1:, :, :] - prediction_padded[:, :, :-1])[..., 1:-1, 1:-1]
    dx = (prediction_padded[:, :, :, 1:] - prediction_padded[:, :, :, :-1])[..., 1:, :, 1:-1]
    dy = (prediction_padded[:, :, :, :, 1:] - prediction_padded[:, :, :, :, :-1])[..., 1:, 1:-1, :]
    dxx = dx[:, :, :, 1:] - dx[:, :, :, -1:]
    dyy = dy[:, :, :, :, 1:] - dy[:, :, :, :, -1:]
    dx = dx[:, :, :, :-1]
    dy = dy[:, :, :, :, :-1]
    l1 = dt + u * dx + v * dy - 1 / Re * (dxx + dyy)
    l2 = dt + u * dx + v * dy - 1 / Re * (dxx + dyy)

    # set nan to 0
    all_losses = (l1**2 + l2**2).sum(axis=1)
    total_size = all_losses.nelement()
    num_nans = torch.isnan(all_losses).sum().item()
    nans = torch.isnan(all_losses)
    all_losses[nans] = 0

    mse_loss = 0.5 * F.mse_loss(all_losses, torch.zeros_like(all_losses), reduction="sum")
    res = mse_loss / (total_size - num_nans)
    if get_all_cells:
        all_losses[nans] = math.nan
        return res, all_losses
    return res


def get_metrics(improved_fc, hc, initial_fc):
    metrics = {}
    magn_sq_init = ((initial_fc - hc) ** 2).sum(axis=1)
    magn_squared_improved = ((improved_fc - hc) ** 2).sum(axis=1)

    metrics["rmse_initial"] = torch.sqrt(magn_sq_init.mean(axis=(-1, -2, -3))).mean().item()
    metrics["rmse_improved"] = (
        torch.sqrt(magn_squared_improved.mean(axis=(-1, -2, -3))).mean().item()
    )
    metrics["rmse_ratio"] = (
        (
            torch.sqrt(magn_squared_improved.mean(axis=(-1, -2, -3)))
            / (torch.sqrt(magn_sq_init.mean(axis=(-1, -2, -3))) + 1e-6)
        )
        .mean()
        .item()
    )

    metrics["evm_initial"] = torch.sqrt(magn_sq_init).mean().item()
    metrics["evm_improved"] = torch.sqrt(magn_squared_improved).mean().item()
    metrics["evm_ratio"] = (
        (
            torch.sqrt(magn_squared_improved).mean(axis=(-1, -2, -3))
            / (torch.sqrt(magn_sq_init).mean(axis=(-1, -2, -3)) + 1e-6)
        )
        .mean()
        .item()
    )

    metrics["r2"] = (
        (
            1
            - (
                (magn_squared_improved.sum(axis=(-1, -2, -3)))
                / (magn_sq_init.sum(axis=(-1, -2, -3)))
            )
        )
        .mean()
        .item()
    )

    metrics["ratio_per_tile"] = (
        (magn_squared_improved.sum(axis=(-1, -2)) / magn_sq_init.sum(axis=(-1, -2)))
        .mean(axis=(0, 1))
        .item()
    )

    return metrics


def loss_function(prediction, target, _lambda=0):
    # dimensions prediction & target: [batch_size, currents, time, lon, lat]

    magnitude_squared_improved = ((prediction - target) ** 2).sum(axis=1)
    loss_rmse = torch.sqrt(magnitude_squared_improved.mean(axis=(-1, -2))).sum()

    if not _lambda:
        return loss_rmse, loss_rmse.item(), 0
    else:
        loss_conservation = compute_conservation_mass_loss(prediction)
        physical_loss = loss_conservation
        return (
            (1 - _lambda) * loss_rmse + _lambda * physical_loss,
            loss_rmse.item(),
            physical_loss.item(),
        )


def get_ratio_accuracy(output_NN, forecast, target) -> Tuple[float, list[float]]:
    assert output_NN.shape == forecast.shape == target.shape
    # Dimensions: batch x currents x time x lon x lat
    magnitude_NN = torch.sqrt(((output_NN - target) ** 2).nansum(axis=[1, 2, 3, 4]))
    magnitude_initial = torch.sqrt(((forecast - target) ** 2).nansum(axis=[1, 2, 3, 4]))
    if (magnitude_initial == 0).sum():
        print("removing nans", (magnitude_initial == 0).sum())
        mask = magnitude_initial != 0
        magnitude_NN = magnitude_NN[mask]
        magnitude_initial = magnitude_initial[mask]
    all_ratios = magnitude_NN / magnitude_initial

    return all_ratios.mean().item(), all_ratios.tolist()


def get_ratio_accuracy_corrected(output_NN, forecast, target) -> Tuple[float, list[float]]:
    assert output_NN.shape == forecast.shape == target.shape
    # Dimensions: batch 0 x currents 1 x time 2 x lon 3 x lat 4
    magnitude_NN = torch.sqrt(((output_NN - target) ** 2).nansum(axis=[1, 3, 4]))
    magnitude_initial = torch.sqrt(((forecast - target) ** 2).nansum(axis=[1, 3, 4]))
    if (magnitude_initial == 0).sum():
        print("removing nans", (magnitude_initial == 0).sum())
        f = magnitude_initial != 0
        magnitude_NN = magnitude_NN[f]
        magnitude_initial = magnitude_initial[f]

    # Mean over time
    all_ratios = np.sqrt((magnitude_NN / magnitude_initial)).mean(axis=-1)

    return all_ratios.mean().item(), all_ratios.tolist()


def get_optimizer(model, name: str, args_optimizer: dict[str, Any], lr: float):
    args_optimizer["lr"] = lr
    if name.lower() == "adam":
        return optim.Adam(model.parameters(), **args_optimizer)
    raise warn("No optimizer!")
    return None


def get_scheduler(cfg_scheduler, optimizer) -> Tuple[optim.lr_scheduler._LRScheduler, bool]:
    name = cfg_scheduler.get("name", "")
    if name.lower() == "reducelronplateau":
        print(f"arguments scheduler: {cfg_scheduler}")
        return (
            optim.lr_scheduler.ReduceLROnPlateau(optimizer, **cfg_scheduler.get("parameters", {})),
            True,
        )
    raise warn("No scheduler!")
    return None, False


def get_model(model_type, cfg_neural_network, device):
    path_weights = cfg_neural_network.pop("path_weights", None)

    if model_type == "mlp":
        model = OceanCurrentMLP(**cfg_neural_network)
    elif model_type == "cnn":
        model = OceanCurrentCNNSubgrid(**cfg_neural_network)
    elif model_type == "rnn":
        model = OceanCurrentRNN(**cfg_neural_network)
    elif model_type == "unetlstm":
        model = OceanCurrentUnetLSTM(**cfg_neural_network)
    elif model_type == "convlstm":
        model = OceanCurrentConvLSTM(**cfg_neural_network)
    else:
        raise Exception("invalid model type provided.")

    # Loading the parameters
    if path_weights is not None:
        print(f"loading model parameter from {path_weights}")
        try:
            model.load_state_dict(torch.load(path_weights, map_location=device))
        except FileNotFoundError:
            # get it from the package distribution
            import ocean_navigation_simulator
            package_path = os.path.dirname(os.path.abspath(ocean_navigation_simulator.__file__))
            model.load_state_dict(torch.load(package_path + path_weights, map_location=device))
        model.eval()

    return model.to(device)


def __create_histogram(list_ratios: List[float], epoch, args, is_training, n_bins=30):
    legend_name = "training" if is_training else "validation"
    plt.figure()
    list_ratios = np.array(list_ratios)
    list_ratios[list_ratios == np.inf] = 100
    plt.hist(list_ratios, bins=n_bins)

    plt.axvline(x=1, color="b", label="x=1")
    plt.title(
        f"Histogram for {legend_name} at epoch {epoch} with mean {list_ratios.mean():.2f}, std: {list_ratios.std():.2f}"
    )
    plt.xlabel("ratio rmse(NN)/rmse(FC)")
    plt.ylabel(f"frequency (over {len(list_ratios)} samples)")
    now = datetime.now()
    folder = os.path.abspath(
        args.get("folder_figure", "./")
        + args["model_type"]
        + "/"
        + now.strftime("%d-%m-%Y_%H-%M-%S")
        + "/"
    )
    filename = (
        f'epoch{epoch}_{legend_name}_loss{(f"{list_ratios.mean():.2f}").replace(".", "_")}.png'
    )
    os.makedirs(folder, exist_ok=True)

    plt.savefig(os.path.join(folder, filename))
    plt.close()


def loop_train_validation(
    training_mode: bool,
    args,
    model,
    device,
    create_histogram_plots,
    data_loader: torch.utils.data.DataLoader,
    epoch: int,
    model_error: bool,
    cfg_dataset: dict[str, any],
    optimizer: Optional[torch.optim.Optimizer] = None,
    suffix="",
):
    legend = ("_training" if training_mode else "_validation") + suffix
    if training_mode:
        model.train()
    else:
        model.eval()

    total_loss_pinn = 0
    total_loss_hindcast = 0
    total_loss_overall = 0
    initial_loss_pinn = 0
    initial_losses_no_pinn = 0
    initial_loss_overall = 0

    list_ratios = list()
    with (torch.autograd.set_detect_anomaly(True) if training_mode else torch.no_grad()):
        with tqdm(data_loader, unit="batch") as tepoch:
            official_metrics = defaultdict(int)
            print("tepoch:", tepoch)
            for tuple_from_dataloader in tepoch:
                if args.return_GP_FC_IMP_FC:
                    data, target, fc, imp_fc = tuple_from_dataloader
                    fc, imp_fc = fc.to(device, dtype=torch.float), imp_fc.to(
                        device, dtype=torch.float
                    )
                else:
                    data, target = tuple_from_dataloader
                if (data, target) == (None, None):
                    continue
                data, target = data.to(device, dtype=torch.float), target.to(
                    device, dtype=torch.float
                )

                # We take the matching input timesteps with the output timesteps
                axis_time = cfg_dataset["index_axis_time"]
                shift_input = cfg_dataset.get("shift_window_input", 0)
                data_same_time = torch.moveaxis(
                    torch.moveaxis(data, axis_time, 0)[
                        shift_input : shift_input + target.shape[axis_time]
                    ],
                    0,
                    axis_time,
                )
                axis_channel = cfg_dataset.get("index_axis_channel", 1)
                indices_chanels_initial_fc = cfg_dataset.get("indices_chanels_initial_fc", None)
                if indices_chanels_initial_fc is not None:
                    data_same_time = torch.moveaxis(
                        torch.moveaxis(data_same_time, axis_channel, 0)[indices_chanels_initial_fc],
                        0,
                        axis_channel,
                    )

                if training_mode:
                    optimizer.zero_grad()
                output = model(data)
                if model_error:
                    output = data_same_time - output

                # Compute the loss
                total_loss_batch, loss_hindcast, loss_pinn = loss_function(
                    output, target, args.lambda_physical_loss
                )
                # gp_loss_batch, gp_loss_hindcast, gp_loss_pinn = loss_function(imp_fc, target, args.lambda_physical_loss)
                # we computed the mean over timesteps and batches
                mean_loss_batch = total_loss_batch / len(output) / output.shape[2]
                for k, v in get_metrics(output, target, data_same_time, not training_mode).items():
                    # if not len(official_metrics):
                    #     official_metrics["magnitudes_per_location" + legend] = np.zeros(output.shape[-2:])
                    official_metrics[k + legend] += v
                if args.return_GP_FC_IMP_FC:
                    for k, v in get_metrics(imp_fc, target, fc).items():
                        official_metrics[k + "_GP" + legend] += v

                official_metrics["count"] += 1

                init_total_loss, init_loss_hindcast, init_loss_pinn = loss_function(
                    data_same_time, target, args.lambda_physical_loss
                )
                initial_losses_no_pinn += init_loss_hindcast
                initial_loss_pinn += init_loss_pinn
                initial_loss_overall += init_total_loss.item()

                total_loss_pinn += loss_pinn
                total_loss_hindcast += loss_hindcast
                total_loss_overall += mean_loss_batch.item()
                ratio, all_ratios = get_ratio_accuracy(output, data_same_time, target)

                list_ratios += all_ratios
                if training_mode:
                    total_loss_batch.backward()
                    optimizer.step()

                # tepoch.set_postfix(loss=str(round(loss_with_pinn.item(), 2)), mean_ratio=str(round(ratio, 2)),
                #                    loss_pinn=str(round(loss_without_pinn.item(), 2)))
                tepoch.set_postfix(loss=str(round(mean_loss_batch.item(), 3)))
        total_loss_overall /= len(data_loader)
        total_loss_pinn /= len(data_loader)
        total_loss_hindcast /= len(data_loader)
        if create_histogram_plots:
            __create_histogram(all_ratios, epoch, args, True)
        print(
            f"{'Training' if training_mode else 'Validation'} avg loss: {total_loss_overall:.2f}"
            + f"  % of ratios <= 1: {((np.array(list_ratios) <= 1).sum() / len(list_ratios) * 100):.2f}%,"
            + f" mean ratio{(np.array(list_ratios).mean()):.2f}"
            + f"Pinn loss: {total_loss_pinn} Hindcast loss: {total_loss_hindcast}"
        )
        total = official_metrics.pop("count")

        for k in official_metrics.keys():
            official_metrics[k] /= total

        return (
            total_loss_overall,
            total_loss_pinn,
            total_loss_hindcast,
            np.array(list_ratios).mean(),
            official_metrics,
        )


def end_training(
    model,
    args,
    train_losses_no_pinn,
    train_losses_pinn,
    validation_losses_no_pinn,
    validation_losses_pinn,
    train_ratios,
    validation_ratios,
):
    train_losses_no_pinn = np.array(train_losses_no_pinn)
    train_losses_pinn = np.array(train_losses_pinn)
    validation_losses_no_pinn = np.array(validation_losses_no_pinn)
    validation_losses_pinn = np.array(validation_losses_pinn)
    train_ratios = np.array(train_ratios)
    validation_ratios = np.array(validation_ratios)

    print(
        f"Training over. Best validation loss {validation_ratios.min()} at epoch {validation_ratios.argmin()}\n"
        f" with losses train pinn:{train_losses_pinn[validation_ratios.argmin()]} test:{validation_losses_pinn[validation_ratios.argmin()]}.\n"
        f" with losses train no pinn:{train_losses_no_pinn[validation_ratios.argmin()]} test:{validation_losses_no_pinn[validation_ratios.argmin()]}.\n"
        f" with train ratio: {train_ratios}.\n"
        f" List of all the training losses with pinn: {train_losses_pinn}"
    )

    if args.save_model:
        torch.save(model.state_dict(), f"{args.model_type}.pt")

    # Log the summary metric using the test set
    wandb.summary["best_validation_ratio"] = validation_ratios.min()
    wandb.finish()


def recursive_doc_dict(dict_input):
    if type(dict_input) is not dict:
        return dict_input
    return DotDict({k: recursive_doc_dict(v) for k, v in dict_input.items()})


def get_args(all_cfgs):
    args = all_cfgs.get("arguments_model_runner", {})
    args.setdefault("batch_size", 64)
    args.setdefault("validation_batch_size", 999)
    args.setdefault("epochs", 14)
    args.setdefault("lr", 1.0)
    args.setdefault("gamma", 0.7)
    args.setdefault("yaml_file_datasets", "")
    args.setdefault("no_cuda", False)
    args.setdefault("dry_run", False)
    args.setdefault("silicon", False)
    args.setdefault("seed", 1)
    args.setdefault("log_interval", 10)
    args.setdefault("save_model", False)
    args.setdefault("max_batches_training_set", -1)
    args.setdefault("max_batches_validation_set", -1)
    args.setdefault("lambda_physical_loss", 0)
    args.setdefault("tags_wandb", [])
    args.setdefault("validate_each_file_separately", True)
    args.setdefault("return_GP_FC_IMP_FC", False)
    args.setdefault("model_to_load", None)
    return recursive_doc_dict(args)
    # END ALTERNATIVE


def main(
    setup_wandb_parameters_sweep: Optional[bool] = False,
    evaluate_only: Optional[bool] = False,
    enable_wandb: Optional[bool] = True,
    model_to_load: Optional[str] = None,
    json_model_from_wandb: Optional[str] = None,
    testing_folder: Optional[str] = None,
    create_histogram_plots: Optional[bool] = False,
):
    np.set_printoptions(precision=2)

    # Load the config file and the dicts
    parser = argparse.ArgumentParser(description="yaml config file path")
    parser.add_argument(
        "--file-configs", type=str, help="name file config to run (without the extension)"
    )
    config_file = parser.parse_args().file_configs + ".yaml"
    all_cfgs = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    args = get_args(all_cfgs)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    cfg_model = all_cfgs.get("model", {})
    cfg_neural_network = recursive_doc_dict(
        cfg_model.get("cfg_neural_network", {}) | {"device": device}
    )
    cfg_dataset = recursive_doc_dict(cfg_model.get("cfg_dataset", {}))
    cfg_optimizer = recursive_doc_dict(args.get("optimizer", {}))
    cfg_scheduler = recursive_doc_dict(args.get("scheduler", {}))
    cfg_data_generation = recursive_doc_dict(all_cfgs.get("data_generation", {}))

    if enable_wandb:
        init_wandb(args, cfg_neural_network, config_file, setup_wandb_parameters_sweep)

    # Load the fields
    torch.manual_seed(args.seed)
    if args.silicon:
        device = torch.device("mps")
    model_error = cfg_model.get("model_error", True)
    print("device:", device)
    print("The Model will predict the " + ("error" if model_error else "hindcast") + ".")
    train_loader, validation_loaders = load_datasets_training_and_validation(
        args, cfg_data_generation, evaluate_only, testing_folder, use_cuda
    )

    # Load model, optimizer and scheduler
    if json_model_from_wandb is not None:
        load_params_from_wandb_json(cfg_neural_network, json_model_from_wandb)
    model = get_model(args.model_type, cfg_neural_network, device)
    load_model_weights_if_necessary(args, model, model_to_load)
    optimizer = get_optimizer(
        model, cfg_optimizer.get("name", ""), cfg_optimizer.get("parameters", {}), args.lr
    )
    scheduler, scheduler_step_takes_argument = get_scheduler(cfg_scheduler, optimizer)

    # initialize lists for loss tracking
    max_loss = math.inf
    train_ratios, validation_ratios = list(), list()
    train_losses_overall, validation_losses_overall = list(), list()
    train_losses_no_pinn, validation_losses_no_pinn = list(), list()
    train_losses_pinn, validation_losses_pinn = list(), list()

    if evaluate_only:
        args.epochs = 1
    for epoch in range(1, args.epochs + 1):
        all_metrics = {}
        # Training
        if not evaluate_only:
            all_metrics = run_training_epoch(
                all_metrics,
                args,
                cfg_dataset,
                create_histogram_plots,
                device,
                epoch,
                model,
                model_error,
                optimizer,
                train_loader,
                train_losses_no_pinn,
                train_losses_overall,
                train_losses_pinn,
                train_ratios,
            )

        # Testing
        loss_no_pinn, loss_pinn, merged_metrics, overall_loss, ratio = run_validation_epoch(
            args,
            cfg_dataset,
            create_histogram_plots,
            device,
            epoch,
            model,
            model_error,
            validation_loaders,
            validation_losses_no_pinn,
            validation_losses_overall,
            validation_losses_pinn,
            validation_ratios,
        )

        all_metrics = add_metrics_to_all_metrics(
            all_metrics,
            loss_no_pinn,
            loss_pinn,
            merged_metrics,
            optimizer,
            overall_loss,
            ratio,
            validation_loaders,
        )
        scheduler_step(optimizer, overall_loss, scheduler, scheduler_step_takes_argument)
        if max_loss > overall_loss.mean():
            max_loss = overall_loss.mean()

            if args.save_model:
                save_model(enable_wandb, epoch, model)
        if enable_wandb:
            wandb.log(all_metrics)

    end_training(
        model,
        args,
        train_losses_no_pinn,
        train_losses_pinn,
        validation_losses_no_pinn,
        validation_losses_pinn,
        train_ratios,
        validation_ratios,
    )

    return all_metrics


def save_model(enable_wandb, epoch, model):
    name_file = f"model_{epoch}.h5"
    print(f"saved model at epoch {epoch}: {name_file}")
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, name_file))
    if enable_wandb:
        wandb.save(name_file)


def add_metrics_to_all_metrics(
    all_metrics,
    loss_no_pinn,
    loss_pinn,
    merged_metrics,
    optimizer,
    overall_loss,
    ratio,
    validation_loaders,
):
    all_metrics |= {"learning rate": optimizer.param_groups[0]["lr"]}
    if len(overall_loss) > 1:
        for i in range(len(overall_loss)):
            all_metrics |= {
                "validation_loss_{i}": overall_loss[i],
                "validation_loss_pinn_{i}": loss_pinn[i],
                "validation_loss_no_pinn_{i}": loss_no_pinn[i],
                "validation_ratio": ratio[i],
            }
    all_metrics |= {
        "validation_loss": overall_loss.mean(),
        "validation_loss_pinn": loss_pinn.mean(),
        "validation_loss_no_pinn": loss_no_pinn.mean(),
        "validation_ratio": ratio.mean(),
    }
    if len(validation_loaders) > 1:
        merged_metrics |= {
            k: v / len(validation_loaders)
            for k, v in merged_metrics.items()
            if k.endswith("_validation")
        }
        all_metrics |= merged_metrics
    return all_metrics


def scheduler_step(optimizer, overall_loss, scheduler, scheduler_step_takes_argument):
    if scheduler is not None:
        if scheduler_step_takes_argument:
            scheduler.step(overall_loss.mean())
            print(f"current lr: {optimizer.param_groups[0]['lr']}")
        else:
            scheduler.step()


def run_validation_epoch(
    args,
    cfg_dataset,
    create_histogram_plots,
    device,
    epoch,
    model,
    model_error,
    validation_loaders,
    validation_losses_no_pinn,
    validation_losses_overall,
    validation_losses_pinn,
    validation_ratios,
):
    print(f"starting Testing epoch {epoch}/{args.epochs}.")
    time.sleep(0.2)
    overall_loss, loss_pinn, loss_no_pinn, ratio = [], [], [], []
    merged_metrics = defaultdict(int)
    for i, validation_loader in enumerate(validation_loaders):
        suffix = f"_{i}"
        overall, pinn, no_pinn, ratios, metrics = loop_train_validation(
            False,
            args,
            model,
            device,
            create_histogram_plots,
            validation_loader,
            epoch,
            model_error,
            cfg_dataset,
            suffix=suffix,
        )
        # Merge all the validations sets into a general one
        for k, v in metrics.items():
            merged_metrics[k[: -len(suffix)]] += v
        overall_loss.append(overall), loss_pinn.append(pinn), loss_no_pinn.append(
            no_pinn
        ), ratio.append(ratios)
        merged_metrics |= metrics
    overall_loss, loss_pinn, = np.array(
        overall_loss
    ), np.array(loss_pinn)
    loss_no_pinn, ratio = np.array(loss_no_pinn), np.array(ratio)
    validation_losses_overall.append(overall_loss)
    validation_losses_pinn.append(loss_pinn)
    validation_losses_no_pinn.append(loss_no_pinn)
    validation_ratios.append(ratio)
    return loss_no_pinn, loss_pinn, merged_metrics, overall_loss, ratio


def run_training_epoch(
    all_metrics,
    args,
    cfg_dataset,
    create_histogram_plots,
    device,
    epoch,
    model,
    model_error,
    optimizer,
    train_loader,
    train_losses_no_pinn,
    train_losses_overall,
    train_losses_pinn,
    train_ratios,
):
    print(f"starting Training epoch {epoch}/{args.epochs}.")
    time.sleep(0.1)
    overall_loss, loss_pinn, loss_no_pinn, ratio, metrics = loop_train_validation(
        True,
        args,
        model,
        device,
        create_histogram_plots,
        train_loader,
        epoch,
        model_error,
        cfg_dataset,
        optimizer,
    )
    all_metrics |= metrics
    train_losses_overall.append(overall_loss)
    train_losses_no_pinn.append(loss_no_pinn)
    train_losses_pinn.append(loss_pinn)
    train_ratios.append(ratio)
    all_metrics |= {
        "train_loss": overall_loss,
        "train_loss_pinn": loss_pinn,
        "train_loss_hc": loss_no_pinn,
        "train_ratio": ratio,
    }
    return all_metrics


def load_datasets_training_and_validation(
    args, cfg_data_generation, evaluate_only, testing_folder, use_cuda
):
    folder_training = cfg_data_generation["parameters_input"]["folder_training"]
    if isinstance(folder_training, str):
        folder_training = [folder_training]
    folder_validation = cfg_data_generation["parameters_input"]["folder_validation"]
    if isinstance(folder_validation, str):
        folder_validation = [folder_validation]
    if folder_training == folder_validation:
        warn("Training and validation use the same dataset!")
    train_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": cfg_data_generation["parameters_input"].get("shuffle_training", True),
    }
    validation_kwargs = {
        "batch_size": args.validation_batch_size,
        "shuffle": cfg_data_generation["parameters_input"].get("shuffle_validation", True),
    }
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True}
        train_kwargs.update(cuda_kwargs)
        validation_kwargs.update(cuda_kwargs)
    if not evaluate_only:
        dataset_training = CustomOceanCurrentsFromFiles(
            folder_training,
            max_items=args.batch_size * args.max_batches_training_set,
            tile_size=args.tile_size,
            return_GP_FC_IMP_FC=args.return_GP_FC_IMP_FC,
        )
        train_loader = torch.utils.data.DataLoader(
            dataset_training, collate_fn=collate_fn, **train_kwargs
        )
    if testing_folder is not None:
        folder_validation = testing_folder
        print("Validation replace by the testing set.")
    # Manage one or multiple validation files evaluation
    if not args.validate_each_file_separately:
        datasets_validation = CustomOceanCurrentsFromFiles(
            folder_validation,
            max_items=args.validation_batch_size * args.max_batches_validation_set,
            tile_size=args.tile_size,
            return_GP_FC_IMP_FC=args.return_GP_FC_IMP_FC,
        )
    else:
        datasets_validation = []
        for folder in folder_validation:
            for path in sorted(os.listdir(folder)):
                if path.endswith("_x.npy"):
                    path = os.path.join(folder, path[: -len("_x.npy")])
                    datasets_validation.append(
                        CustomOceanCurrentsFromFiles(
                            max_items=args.validation_batch_size * args.max_batches_validation_set,
                            tile_size=args.tile_size,
                            filename_with_path=path,
                            return_GP_FC_IMP_FC=args.return_GP_FC_IMP_FC,
                        )
                    )
    print(
        f"lengths ds: training:{len(dataset_training) if not evaluate_only else 'NA'}, validation:{[len(dataset_validation) for dataset_validation in datasets_validation]}"
    )
    validation_loaders = [
        torch.utils.data.DataLoader(dataset_validation, collate_fn=collate_fn, **validation_kwargs)
        for dataset_validation in datasets_validation
    ]
    return train_loader, validation_loaders


def load_model_weights_if_necessary(args, model, model_to_load):
    if args.model_to_load is not None:
        model.load_state_dict(torch.load(args.model_to_load))
    if model_to_load is not None:
        if args.model_to_load is not None:
            raise UserWarning(
                "The weights specified in the config file is not loaded. The weights specified as a parameter is loaded instead"
            )
        model.load_state_dict(torch.load(model_to_load))


def load_params_from_wandb_json(cfg_neural_network, json_model_from_wandb):
    file = open(json_model_from_wandb)
    data = recursive_doc_dict(json.load(file))
    cfg_neural_network.ch_sz = data.ch_sz.value
    cfg_neural_network.downsizing_method = data.downsizing_method.value
    cfg_neural_network.dropout_encoder = data.dropout_encoder.value
    cfg_neural_network.dropout_bottom = data.dropout_bottom.value
    cfg_neural_network.dropout_decoder = data.dropout_decoder.value
    cfg_neural_network.activation = data.activation.value


def init_wandb(args, cfg_neural_network, config_file, setup_wandb_parameters_sweep):
    wandb.init(project="Seaweed_forecast_improvement", entity="killian2k", tags=args.tags_wandb)
    print(f"starting run: {wandb.run.name}")
    os.environ["WANDB_NOTEBOOK_NAME"] = "Seaweed_forecast_improvement"
    wandb.config.update(args, allow_val_change=False)
    wandb.config.update(cfg_neural_network, allow_val_change=False)
    wandb.save(config_file)
    if setup_wandb_parameters_sweep:
        setup_vars_from_wandb(args, cfg_neural_network)


def setup_vars_from_wandb(args, cfg_neural_network):
    args.epochs = wandb.config.epochs
    args.batch_size = wandb.config.batch_size
    args.lr = wandb.config.lr
    args.optimizer.parameters.weight_decay = wandb.config.weight_decay
    cfg_neural_network.ch_sz = wandb.config.ch_sz
    cfg_neural_network.downsizing_method = wandb.config.downsizing_method
    cfg_neural_network.dropout_encoder = wandb.config.dropout_encoder
    cfg_neural_network.dropout_bottom = wandb.config.dropout_bottom
    cfg_neural_network.dropout_decoder = wandb.config.dropout_decoder
    cfg_neural_network.activation = wandb.config.activation


if __name__ == "__main__":
    main()
