from __future__ import print_function

import argparse
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
from ocean_navigation_simulator.ocean_observer.models.CustomOceanCurrentsFromFiles import CustomOceanCurrentsFromFiles
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentCNN import OceanCurrentCNNSubgrid
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentConvLSTM import OceanCurrentConvLSTM
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentUnetLSTM import OceanCurrentUnetLSTM
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentsMLP import OceanCurrentMLP
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentsRNN import OceanCurrentRNN

now = datetime.now()


def collate_fn(batch):
    batch_filtered = list(filter(lambda x: x[0] is not None, batch))
    if not len(batch_filtered):
        return None, None
    return torch.utils.data.dataloader.default_collate(batch_filtered)


# def compute_burger_loss(input, u, v):
#     # value r_e from paper by Taco de Wolff
#     Re = math.pi / 0.01
#
#     u_t = grad(u, t, create_graph=True, grad_outputs=torch.ones_like(u))[0]
#     v_t = grad(v, t, create_graph=True, grad_outputs=torch.ones_like(v))[0]
#     u_x = grad(u, x, create_graph=True, grad_outputs=torch.ones_like(u))[0]
#     v_x = grad(v, x, create_graph=True, grad_outputs=torch.ones_like(v))[0]
#     u_xx = grad(u_x, x, create_graph=True, grad_outputs=torch.ones_like(u_x))[0]
#     v_xx = grad(v_x, x, create_graph=True, grad_outputs=torch.ones_like(v_x))[0]
#     u_y = grad(u, y, create_graph=True, grad_outputs=torch.ones_like(u))[0]
#     v_y = grad(v, y, create_graph=True, grad_outputs=torch.ones_like(v))[0]
#     u_yy = grad(u_y, y, create_graph=True, grad_outputs=torch.ones_like(u_y))[0]
#     v_yy = grad(v_y, y, create_graph=True, grad_outputs=torch.ones_like(v_y))[0]
#     p1 = u_t + u * u_x + v * v_y - 1 / Re * (u_xx + u_yy)
#     p2 = v_t + u * v_x + v * v_y - 1 / Re * (v_xx + v_yy)
#     return p1 + p2

# def f(self, x, t, u):
#     u_t = grad(u, t, create_graph=True, grad_outputs=torch.ones_like(u))[0]
#     u_x = grad(u, x, create_graph=True, grad_outputs=torch.ones_like(u))[0]
#     u_xx = grad(u_x, x, create_graph=True, grad_outputs=torch.ones_like(u_x))[0]
#
#     return u_t + self.lambda1 * u * u_x - self.lambda2 * u_xx

sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize',
        'name': 'validation_r2_validation_merged'
    },
    'parameters': {
        'batch_size': {'values': [16, 32, 64, 128]},
        'epochs': {'values': [35]},
        'lr': {'max': 0.01, 'min': 0.00001},
        'model_error': {'values': [True, False]},
        'ch_sz': {'values': [[6, 18, 36, 60, 112], [6, 24, 48, 96, 192], [6, 32, 64, 64, 128], [6, 32, 64, 128, 256]]},
        'downsizing_method': {'values': ['maxpool', 'conv', 'avgpool']},
        'dropout_encoder': {'distribution': 'uniform', 'max': 0.3, 'min': 0.01},
        'dropout_bottom': {'distribution': 'uniform', 'max': 0.8, 'min': 0.01},
        'dropout_decoder': {'distribution': 'uniform', 'max': 0.8, 'min': 0.01},
        'weight_decay': {'distribution': 'uniform', 'min': 0.00001, 'max': 0.01},
    }
}


# sweep_id = wandb.sweep(sweep=sweep_configuration, project='Seaweed_forecast_improvement')


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
    #
    # top_left[:, [1]] *= -1
    # top_right[:, [0, 1]] *= -1
    # bottom_right[:, [0]] *= -1
    top_left = pred[..., :-1, :-1]
    top_right = pred[..., :-1, 1:]
    bottom_left = pred[..., 1:, :-1]
    bottom_right = pred[..., 1:, 1:]
    all_losses = (-top_left[:, [1]] + top_left[:, [0]] - top_right +
                  bottom_left - bottom_right[:, [0]] + bottom_right[:, [1]]).sum(axis=1)

    # set nans to 0
    total_size = all_losses.nelement()
    num_nans = torch.isnan(all_losses).sum().item()
    nans = torch.isnan(all_losses)
    all_losses[nans] = 0
    res = 0.5 * (F.mse_loss(all_losses, torch.zeros_like(all_losses), reduction='sum') / (total_size - num_nans))
    if get_all_cells:
        all_losses[nans] = math.nan
        return res, all_losses
    return res


def compute_burgers_loss(prediction, Re=math.pi / 0.01, get_all_cells=True):
    # Add the boundaries
    X, Y = prediction.shape[-2:]
    batches = len(prediction)
    boundary_u_t0 = torch.tensor(
        [[[math.sin(math.pi * x / X) * math.cos(math.pi * y / Y) for y in range(Y)] for x in range(X)]]).expand(batches,
                                                                                                                -1, -1)
    boundary_v_t0 = torch.tensor(
        [[[math.cos(math.pi * x / X) * math.sin(math.pi * y / Y) for y in range(Y)] for x in range(X)]]).expand(batches,
                                                                                                                -1, -1)
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
    all_losses = (l1 ** 2 + l2 ** 2).sum(axis=1)
    total_size = all_losses.nelement()
    num_nans = torch.isnan(all_losses).sum().item()
    nans = torch.isnan(all_losses)
    all_losses[nans] = 0

    mse_loss = 0.5 * F.mse_loss(all_losses, torch.zeros_like(all_losses), reduction='sum')
    res = mse_loss / (total_size - num_nans)
    if get_all_cells:
        all_losses[nans] = math.nan
        return res, all_losses
    return res


def get_metrics(improved_fc, hc, initial_fc, print_r2=False):
    metrics = {}
    magn_sq_init = ((initial_fc - hc) ** 2).sum(axis=1)
    magn_sq_impr = ((improved_fc - hc) ** 2).sum(axis=1)

    # metrics["magnitudes_per_location"] = (
    #         magn_sq_impr.sum(axis=(0, 1)) / magn_sq_init.sum(axis=(0, 1))).cpu().detach().numpy()

    metrics["rmse_initial"] = torch.sqrt(magn_sq_init.mean(axis=(-1, -2, -3))).mean().item()
    metrics["rmse_improved"] = torch.sqrt(magn_sq_impr.mean(axis=(-1, -2, -3))).mean().item()
    metrics["rmse_ratio"] = (torch.sqrt(magn_sq_impr.mean(axis=(-1, -2, -3))) / (
            torch.sqrt(magn_sq_init.mean(axis=(-1, -2, -3))) + 1e-6)).mean().item()

    metrics["evm_initial"] = torch.sqrt(
        magn_sq_init).mean().item()  # np.sqrt(((initial_fc - hc) ** 2).sum(axis=-1)).mean()
    metrics["evm_improved"] = torch.sqrt(magn_sq_impr).mean().item()
    metrics["evm_ratio"] = (torch.sqrt(magn_sq_impr).mean(axis=(-1, -2, -3)) / (
            torch.sqrt(magn_sq_init).mean(axis=(-1, -2, -3)) + 1e-6)).mean().item()

    metrics["r2"] = (1 - ((magn_sq_impr.sum(axis=(-1, -2, -3))) / (magn_sq_init.sum(axis=(-1, -2, -3))))).mean().item()
    if print_r2:
        print("r2s:", (1 - ((magn_sq_impr.sum(axis=(-1, -2, -3))) / (magn_sq_init.sum(axis=(-1, -2, -3))))))

    metrics["ratio_per_tile"] = (magn_sq_impr.sum(axis=(-1, -2)) / magn_sq_init.sum(axis=(-1, -2))).mean(
        axis=(0, 1)).item()

    return metrics


def loss_function(prediction, target, _lambda=0):
    # dimensions prediction & target: [batch_size, currents, time, lon, lat]

    # assert prediction.shape == target.shape and (input is None or input.shape == target.shape)
    # losses = []
    # losses.append(torch.sqrt(F.mse_loss(prediction, target, reduction='mean')))
    #
    # # x FC input current u
    # # y FC input current v
    # # t time of forecasts
    # # u = improved forecast current u
    # # v = improved forecast current u
    # x, y = input[:, 0], target[:, 1]
    # u, v = prediction[:, 0], prediction[:, 0]
    # pinn_loss = compute_burger_loss(x, y, t, u, v)
    #
    # losses.append(torch.sqrt(F.mse_loss(pinn_loss, torch.zeros_like(pinn_loss), reduction='mean')))
    # return losses
    magn_sq_impr = ((prediction - target) ** 2).sum(axis=1)
    loss_rmse = torch.sqrt(magn_sq_impr.mean(axis=(-1, -2))).sum()
    # loss_rmse = torch.sqrt(F.mse_loss(prediction, target, reduction='mean') + 1e-8)
    # loss_r2 = 1 - (prediction - target)

    if not _lambda:
        return loss_rmse, loss_rmse.item(), 0
    else:
        # loss_burger = compute_burgers_loss(prediction)
        loss_conservation = compute_conservation_mass_loss(prediction)
        physical_loss = loss_conservation
        return (1 - _lambda) * loss_rmse + _lambda * physical_loss, loss_rmse.item(), physical_loss.item()


def get_ratio_accuracy(output_NN, forecast, target) -> Tuple[float, list[float]]:
    assert output_NN.shape == forecast.shape == target.shape
    # Dimensions: batch x currents x time x lon x lat
    magn_NN = torch.sqrt(((output_NN - target) ** 2).nansum(axis=[1, 2, 3, 4]))
    magn_initial = torch.sqrt(((forecast - target) ** 2).nansum(axis=[1, 2, 3, 4]))
    if (magn_initial == 0).sum():
        # print(magn_initial.shape, (magn_initial == 0), (magn_initial == 0).sum())
        # raise Exception("Found Nans! Should not have happened")
        print("removing nans", (magn_initial == 0).sum())
        f = (magn_initial != 0)
        magn_NN = magn_NN[f]
        magn_initial = magn_initial[f]
    all_ratios = magn_NN / magn_initial

    return all_ratios.mean().item(), all_ratios.tolist()


def get_ratio_accuracy_corrected(output_NN, forecast, target) -> Tuple[float, list[float]]:
    assert output_NN.shape == forecast.shape == target.shape
    # Dimensions: batch 0 x currents 1 x time 2 x lon 3 x lat 4
    magn_NN = torch.sqrt(((output_NN - target) ** 2).nansum(axis=[1, 3, 4]))
    magn_initial = torch.sqrt(((forecast - target) ** 2).nansum(axis=[1, 3, 4]))
    if (magn_initial == 0).sum():
        # print(magn_initial.shape, (magn_initial == 0), (magn_initial == 0).sum())
        # raise Exception("Found Nans! Should not have happened")
        print("removing nans", (magn_initial == 0).sum())
        f = (magn_initial != 0)
        magn_NN = magn_NN[f]
        magn_initial = magn_initial[f]
    # Mean over time
    all_ratios = np.sqrt((magn_NN / magn_initial)).mean(axis=-1)

    return all_ratios.mean().item(), all_ratios.tolist()


def get_optimizer(model, name: str, args_optimizer: dict[str, Any], lr: float):
    args_optimizer['lr'] = lr
    if name.lower() == "adam":
        return optim.Adam(model.parameters(), **args_optimizer)
    raise warn("No optimizer!")
    return None


def get_scheduler(cfg_scheduler, optimizer) -> Tuple[optim.lr_scheduler._LRScheduler, bool]:
    name = cfg_scheduler.get("name", "")
    if name.lower() == "reducelronplateau":
        print(f"arguments scheduler: {cfg_scheduler}")
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **cfg_scheduler.get("parameters", {})), True
    raise warn("No scheduler!")
    return None, False


def get_model(model_type, cfg_neural_network, device):
    path = cfg_neural_network.pop("path_parameters", None)

    if model_type == 'mlp':
        model = OceanCurrentMLP(**cfg_neural_network)
    elif model_type == 'cnn':
        model = OceanCurrentCNNSubgrid(**cfg_neural_network)
    elif model_type == 'rnn':
        model = OceanCurrentRNN(**cfg_neural_network)
    elif model_type == 'unetlstm':
        model = OceanCurrentUnetLSTM(**cfg_neural_network)
    elif model_type == 'convlstm':
        model = OceanCurrentConvLSTM(**cfg_neural_network)
    else:
        raise Exception("invalid model type provided.")
    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

    return model.to(device)


def __create_loss_plot(args, train_losses, validation_losses, mean_ratio_train, mean_ratio_validation, use_pinn=False):
    # Plot the loss and the LR

    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(train_losses, color='green', marker='o')
    ax.plot(validation_losses, color='red', marker='o')

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss ", color='green')
    ax2 = ax.twinx()
    ax2.plot(mean_ratio_train, color='lightgreen')
    ax2.plot(mean_ratio_validation, color='orangered')
    ax2.set_ylabel("Mean_ratio", color='blue')
    folder = os.path.abspath(
        args.get("folder_figure", "./") + args["model_type"] + "/" + now.strftime("%d-%m-%Y_%H-%M-%S") + "/")
    filename = f"_loss_and_lr_{len(mean_ratio_train)}_{'pinn' if use_pinn else 'no_pinn'}.png"
    os.makedirs(folder, exist_ok=True)
    print(f"saving file {filename} histogram at: {folder}")
    fig.savefig(os.path.join(folder, filename))


def __create_histogram(list_ratios: List[float], epoch, args, is_training, n_bins=30):
    legend_name = "training" if is_training else "validation"
    plt.figure()
    list_ratios = np.array(list_ratios)
    list_ratios[list_ratios == np.inf] = 100
    plt.hist(list_ratios, bins=n_bins)

    plt.axvline(x=1, color='b', label='x=1')
    plt.title(
        f"Histogram for {legend_name} at epoch {epoch} with mean {list_ratios.mean():.2f}, std: {list_ratios.std():.2f}")
    plt.xlabel("ratio rmse(NN)/rmse(FC)")
    plt.ylabel(f"frequency (over {len(list_ratios)} samples)")

    folder = os.path.abspath(
        args.get("folder_figure", "./") + args["model_type"] + "/" + now.strftime("%d-%m-%Y_%H-%M-%S") + "/")
    filename = f'epoch{epoch}_{legend_name}_loss{(f"{list_ratios.mean():.2f}").replace(".", "_")}.png'
    os.makedirs(folder, exist_ok=True)
    # print(f"saving file {filename} histogram at: {folder}")
    plt.savefig(os.path.join(folder, filename))
    plt.close()

    # wandb histogram
    # data = [[i, ratio] for i, ratio in enumerate(list_ratios)]
    # fields = {"x": "sample",
    #           "value": "ratios"}
    # # table = wandb.Table(data=data, columns=fields)
    # # wandb.log({'histogram_ratio_rmses': wandb.plot.histogram(table, "ratios", title="Ratio NN vs FC Distribution")})
    # # Use the table to populate the new custom chart preset
    # # To use your own saved chart preset, change the vega_spec_name
    # my_custom_chart = wandb.plot_table(vega_spec_name="carey/new_chart",
    #                                    data_table=table,
    #                                    fields=fields,
    #                                    )
    # # Log the plot to have it show up in the UI
    # wandb.log({"custom_chart": my_custom_chart})


def loop_train_validation(training_mode: bool, args, model, device, data_loader: torch.utils.data.DataLoader,
                          epoch: int, model_error: bool, cfg_dataset: dict[str, any],
                          optimizer: Optional[torch.optim.Optimizer] = None, print_r2s=False, suffix=""):
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
                    fc, imp_fc = fc.to(device, dtype=torch.float), imp_fc.to(device, dtype=torch.float)
                else:
                    data, target = tuple_from_dataloader
                if (data, target) == (None, None):
                    continue
                data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)

                # We take the matching input timesteps with the output timesteps
                axis_time = cfg_dataset["index_axis_time"]
                shift_input = cfg_dataset.get("shift_window_input", 0)
                data_same_time = torch.moveaxis(
                    torch.moveaxis(data, axis_time, 0)[shift_input:shift_input + target.shape[axis_time]], 0, axis_time)
                axis_channel = cfg_dataset.get("index_axis_channel", 1)
                indices_chanels_initial_fc = cfg_dataset.get("indices_chanels_initial_fc", None)
                if indices_chanels_initial_fc is not None:
                    data_same_time = torch.moveaxis(
                        torch.moveaxis(data_same_time, axis_channel, 0)[indices_chanels_initial_fc], 0, axis_channel)

                if training_mode:
                    optimizer.zero_grad()
                output = model(data)
                if model_error:
                    output = data_same_time - output

                # Compute the loss
                total_loss_batch, loss_hindcast, loss_pinn = loss_function(output, target, args.lambda_physical_loss)
                # gp_loss_batch, gp_loss_hindcast, gp_loss_pinn = loss_function(imp_fc, target, args.lambda_physical_loss)
                # we computed the mean over timesteps and batches
                mean_loss_batch = total_loss_batch / len(output) / output.shape[2]
                for k, v in get_metrics(output, target, data_same_time, not training_mode and print_r2s).items():
                    # if not len(official_metrics):
                    #     official_metrics["magnitudes_per_location" + legend] = np.zeros(output.shape[-2:])
                    official_metrics[k + legend] += v
                if args.return_GP_FC_IMP_FC:
                    for k, v in get_metrics(imp_fc, target, fc).items():
                        official_metrics[k + "_GP" + legend] += v

                official_metrics["count"] += 1

                init_total_loss, init_loss_hindcast, init_loss_pinn = loss_function(data_same_time, target,
                                                                                    args.lambda_physical_loss)
                initial_losses_no_pinn += init_loss_hindcast
                initial_loss_pinn += init_loss_pinn
                initial_loss_overall += init_total_loss.item()

                total_loss_pinn += loss_pinn
                total_loss_hindcast += loss_hindcast
                total_loss_overall += mean_loss_batch.item()
                ratio, all_ratios = get_ratio_accuracy(output,
                                                       data_same_time,
                                                       target)

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
        __create_histogram(all_ratios, epoch, args, True)
        print(
            f"{'Training' if training_mode else 'Validation'} avg loss: {total_loss_overall:.2f}" +
            f"  % of ratios <= 1: {((np.array(list_ratios) <= 1).sum() / len(list_ratios) * 100):.2f}%," +
            f" mean ratio{(np.array(list_ratios).mean()):.2f}" +
            f"Pinn loss: {total_loss_pinn} Hindcast loss: {total_loss_hindcast}")
        total = official_metrics.pop('count')

        for k in official_metrics.keys():
            official_metrics[k] /= total

        return total_loss_overall, total_loss_pinn, total_loss_hindcast, np.array(
            list_ratios).mean(), official_metrics, all_metrics


def end_training(model, args, train_losses_no_pinn, train_losses_pinn, validation_losses_no_pinn,
                 validation_losses_pinn, train_ratios, validation_ratios):
    train_losses_no_pinn = np.array(train_losses_no_pinn)
    train_losses_pinn = np.array(train_losses_pinn)
    validation_losses_no_pinn = np.array(validation_losses_no_pinn)
    validation_losses_pinn = np.array(validation_losses_pinn)
    train_ratios = np.array(train_ratios)
    validation_ratios = np.array(validation_ratios)
    __create_loss_plot(args, train_losses_no_pinn, train_losses_no_pinn, train_ratios, validation_ratios,
                       use_pinn=False)
    __create_loss_plot(args, train_losses_pinn, train_losses_pinn, train_ratios, validation_ratios, use_pinn=True)

    print(
        f"Training over. Best validation loss {validation_ratios.min()} at epoch {validation_ratios.argmin()}\n"
        f" with losses train pinn:{train_losses_pinn[validation_ratios.argmin()]} test:{validation_losses_pinn[validation_ratios.argmin()]}.\n"
        f" with losses train no pinn:{train_losses_no_pinn[validation_ratios.argmin()]} test:{validation_losses_no_pinn[validation_ratios.argmin()]}.\n"
        f" List of all the training losses with pinn: {train_losses_pinn}")

    if args.save_model:
        torch.save(model.state_dict(), f"{args.model_type}.pt")

    # Log the summary metric using the test set
    # wandb.summary['test_accuracy'] = ...
    wandb.summary['best_validation_ratio'] = validation_ratios.min()
    wandb.finish()


def recursive_doc_dict(dict_input):
    if type(dict_input) is not dict:
        return dict_input
    return DotDict({k: recursive_doc_dict(v) for k, v in dict_input.items()})


def get_args(all_cfgs):
    # Training settings
    # parser = argparse.ArgumentParser(description='PyTorch model')
    # parser.add_argument('--batch-size', type=int, default=64, metavar='N',
    #                     help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    # parser.add_argument('--epochs', type=int, default=14, metavar='N',
    #                     help='number of epochs to train (default: 14)')
    # parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
    #                     help='learning rate (default: 1.0)')
    # parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
    #                     help='Learning rate step gamma (default: 0.7)')
    # parser.add_argument('--yaml-file-datasets', type=str, default='',
    #                     help='filname of the yaml file to use to download the data in the folder scenarios/neural_networks')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='disables CUDA training')
    # parser.add_argument('--dry-run', action='store_true', default=False,
    #                     help='quickly check a single pass')
    # parser.add_argument('--silicon', action='store_true', default=False,
    #                     help='enable Mac silicon optimization')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                     help='how many batches to wait before logging training status')
    # parser.add_argument('--save-model', action='store_true', default=False,
    #                     help='For Saving the current Model')
    # parser.add_argument('--model-type', type=str, default='mlp')
    #
    # parser.add_argument('--max-batches-training-set', type=int, default=-1)
    # parser.add_argument('--max-batches-validation-set', type=int, default=-1)
    # args = parser.parse_args()
    # cfgs = yaml.load(open(os.getcwd() + "/config/" + args.file_configs + ".yaml", 'r'), Loader=yaml.FullLoader)

    # ALTERNATIVE:
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


def main(setup_wandb_parameters_sweep: bool = False, evaluate_only: bool = False, enable_wandb: bool = True):
    np.set_printoptions(precision=2)
    parser = argparse.ArgumentParser(description='yaml config file path')
    parser.add_argument('--file-configs', type=str, help='name file config to run (without the extension)')
    config_file = parser.parse_args().file_configs + ".yaml"
    all_cfgs = yaml.load(open(config_file, 'r'),
                         Loader=yaml.FullLoader)
    args = get_args(all_cfgs)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    torch.manual_seed(args.seed)
    device = device
    print("device:", device)
    if args.silicon:
        device = torch.device("mps")

    cfg_model = all_cfgs.get("model", {})
    cfg_neural_network = recursive_doc_dict(cfg_model.get("cfg_neural_network", {}) | {"device": device})
    cfg_dataset = recursive_doc_dict(cfg_model.get("cfg_dataset", {}))
    cfg_optimizer = recursive_doc_dict(args.get("optimizer", {}))
    cfg_scheduler = recursive_doc_dict(args.get("scheduler", {}))
    cfg_data_generation = recursive_doc_dict(all_cfgs.get("data_generation", {}))
    model_error = cfg_model.get("model_error", True)
    print("The Model will predict the " + ("error" if model_error else "hindcast") + ".")
    folder_training = cfg_data_generation["parameters_input"]["folder_training"]
    if isinstance(folder_training, str):
        folder_training = [folder_training]
    folder_validation = cfg_data_generation["parameters_input"]["folder_validation"]
    if isinstance(folder_validation, str):
        folder_validation = [folder_validation]
    if folder_training == folder_validation:
        warn("Training and validation use the same dataset!!!")

    train_kwargs = {'batch_size': args.batch_size,
                    'shuffle': cfg_data_generation['parameters_input'].get('shuffle_training', True)}
    test_kwargs = {'batch_size': args.validation_batch_size,
                   'shuffle': cfg_data_generation['parameters_input'].get('shuffle_validation', True)}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    dataset_training = CustomOceanCurrentsFromFiles(folder_training,
                                                    max_items=args.batch_size * args.max_batches_training_set,
                                                    tile_size=args.tile_size,
                                                    return_GP_FC_IMP_FC=args.return_GP_FC_IMP_FC)
    print("validate_each_file_separately", args.validate_each_file_separately)
    if not args.validate_each_file_separately:
        datasets_validation = CustomOceanCurrentsFromFiles(folder_validation,
                                                           max_items=args.validation_batch_size * args.max_batches_validation_set,
                                                           tile_size=args.tile_size,
                                                           return_GP_FC_IMP_FC=args.return_GP_FC_IMP_FC)
    else:
        datasets_validation = []
        for folder in folder_validation:
            for path in sorted(os.listdir(folder)):
                if path.endswith("_x.npy"):
                    path = os.path.join(folder, path[:-len("_x.npy")])
                    datasets_validation.append(CustomOceanCurrentsFromFiles(
                        max_items=args.validation_batch_size * args.max_batches_validation_set,
                        tile_size=args.tile_size, filename_with_path=path,
                        return_GP_FC_IMP_FC=args.return_GP_FC_IMP_FC))

    print(
        f"lengths ds: training:{len(dataset_training)}, validation:{[len(dataset_validation) for dataset_validation in datasets_validation]}")

    train_loader = torch.utils.data.DataLoader(dataset_training, collate_fn=collate_fn, **train_kwargs)
    validation_loaders = [torch.utils.data.DataLoader(dataset_validation, collate_fn=collate_fn, **test_kwargs) for
                          dataset_validation in datasets_validation]

    wandb.init(project="Seaweed_forecast_improvement", entity="killian2k",
               tags=args.tags_wandb)  # , name=f"experiment_{}")
    print(f"starting run: {wandb.run.name}")
    os.environ['WANDB_NOTEBOOK_NAME'] = "Seaweed_forecast_improvement"
    if enable_wandb:
        wandb.config.update(args, allow_val_change=False)
        wandb.config.update(cfg_neural_network, allow_val_change=False)
        wandb.save(config_file)
        if setup_wandb_parameters_sweep:
            # print("old ch_sz", cfg_neural_network.ch_sz)
            args.epochs = 22
            print("epochs set to ", args.epochs)
            # args.epochs = wandb.config.epochs
            args.batch_size = wandb.config.batch_size
            args.lr = wandb.config.lr
            args.optimizer.parameters.weight_decay = wandb.config.weight_decay
            cfg_neural_network.ch_sz = wandb.config.ch_sz
            cfg_neural_network.downsizing_method = wandb.config.downsizing_method
            cfg_neural_network.dropout_encoder = wandb.config.dropout_encoder
            cfg_neural_network.dropout_bottom = wandb.config.dropout_bottom
            cfg_neural_network.dropout_decoder = wandb.config.dropout_decoder
            cfg_neural_network.activation = wandb.config.activation
            # print("new ch_sz", cfg_neural_network.ch_sz)

    print(f"Weight physical loss: {args.lambda_physical_loss}")
    model = get_model(args.model_type, cfg_neural_network, device)
    if args.model_to_load:
        model.load_state_dict(torch.load(args.model_to_load))
    optimizer = get_optimizer(model, cfg_optimizer.get("name", ""), cfg_optimizer.get("parameters", {}), args.lr)

    # scheduler = schedulers.StepLR(optimizer, step_size=1, gamma=args.gamma)
    scheduler, scheduler_step_takes_argument = get_scheduler(cfg_scheduler, optimizer)
    print(f"optimizer: {optimizer}")

    max_loss = math.inf
    train_ratios, validation_ratios = list(), list()
    train_losses_overall, validation_losses_overall = list(), list()
    train_losses_no_pinn, validation_losses_no_pinn = list(), list()
    train_losses_pinn, validation_losses_pinn = list(), list()

    for epoch in range(1, args.epochs + 1):
        all_metrics = {}
        # Training
        if not evaluate_only:
            print(f"starting Training epoch {epoch}/{args.epochs}.")
            time.sleep(0.2)
            overall_loss, loss_pinn, loss_no_pinn, ratio, metrics, official_metrics = loop_train_validation(True, args,
                                                                                                            model,
                                                                                                            device,
                                                                                                            train_loader,
                                                                                                            epoch,
                                                                                                            model_error,
                                                                                                            cfg_dataset,
                                                                                                            optimizer)
            all_metrics |= metrics
            train_losses_overall.append(overall_loss)
            train_losses_no_pinn.append(loss_no_pinn)
            train_losses_pinn.append(loss_pinn)
            train_ratios.append(ratio)
            metrics |= {"train_loss": overall_loss, "train_loss_pinn": loss_pinn, "train_loss_hc": loss_no_pinn,
                        "train_ratio": ratio}

            all_metrics |= official_metrics

        # Testing
        print(f"starting Testing epoch {epoch}/{args.epochs}.")
        time.sleep(0.2)
        overall_loss, loss_pinn, loss_no_pinn, ratio = [], [], [], []
        merged_metrics = defaultdict(int)
        for i, validation_loader in enumerate(validation_loaders):
            suffix = f"_{i}"
            overall, pinn, no_pinn, ra, metrics, official_metrics = loop_train_validation(False, args, model, device,
                                                                                          validation_loader, epoch,
                                                                                          model_error, cfg_dataset,
                                                                                          suffix=suffix)
            for k, v in metrics.items():
                merged_metrics[k[:-len(suffix)]] += v
            overall_loss.append(overall), loss_pinn.append(pinn), loss_no_pinn.append(no_pinn), ratio.append(ra)
        overall_loss, loss_pinn, loss_no_pinn, ratio = np.array(overall_loss), np.array(loss_pinn), np.array(
            loss_no_pinn), np.array(ratio)
        validation_losses_overall.append(overall_loss)
        validation_losses_pinn.append(loss_pinn)
        validation_losses_no_pinn.append(loss_no_pinn)
        validation_ratios.append(ratio)
        all_metrics |= official_metrics
        all_metrics |= {"learning rate": optimizer.param_groups[0]['lr']}

        if len(overall_loss) > 1:
            for i in range(len(overall_loss)):
                all_metrics |= {f"validation_loss_{i}": overall_loss[i], f"validation_loss_pinn_{i}": loss_pinn[i],
                                f"validation_loss_no_pinn_{i}": loss_no_pinn[i],
                                f"validation_ratio": ratio[i]}

        all_metrics |= {f"validation_loss": overall_loss.mean(), f"validation_loss_pinn": loss_pinn.mean(),
                        f"validation_loss_no_pinn": loss_no_pinn.mean(),
                        f"validation_ratio": ratio.mean()}
        if scheduler is not None:
            if scheduler_step_takes_argument:
                scheduler.step(overall_loss.mean())
                print(f"current lr: {optimizer.param_groups[0]['lr']}")
            else:
                scheduler.step()
        if len(validation_loaders) > 1:
            merged_metrics = {"validation_" + k + "_merged": v / len(validation_loaders) for k, v in
                              merged_metrics.items()}
            print(merged_metrics)
            all_metrics |= merged_metrics
        if max_loss > overall_loss.mean():
            max_loss = overall_loss.mean()

            name_file = f'model_{epoch}.h5'
            print(f"saved model at epoch {epoch}: {name_file}")
            if wandb:
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, name_file))
                wandb.save(name_file)
                wandb.log(all_metrics)

        if epoch % 30 == 0:
            __create_loss_plot(args, train_losses_pinn, validation_losses_pinn, train_ratios, validation_ratios,
                               True)
            __create_loss_plot(args, train_losses_no_pinn, validation_losses_no_pinn, train_ratios,
                               validation_ratios,
                               False)
        # finally:
        #
        #     end_training(model, args, train_losses_no_pinn, train_losses_pinn, validation_losses_no_pinn,
        #                  validation_losses_pinn,
        #                  train_ratios, validation_ratios)
        #     del model, optimizer, scheduler, scheduler_step_takes_argument, max_loss, \
        #         train_ratios, validation_ratios, train_losses_overall, validation_losses_overall
        #     train_losses_pinn, validation_losses_pinn, train_losses_no_pinn, validation_losses_no_pinn, metrics


if __name__ == '__main__':
    main()
