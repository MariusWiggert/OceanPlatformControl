from __future__ import print_function

import argparse
import os
import time
from datetime import datetime
from typing import Tuple, List, Any
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
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentCNN_subgrid_2 import OceanCurrentCNNSubgrid
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentConvLSTM import OceanCurrentConvLSTM
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentUnetLSTM import OceanCurrentUnetLSTM
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentsMLP import OceanCurrentMLP
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentsRNN import OceanCurrentRNN

now = datetime.now()


def collate_fn(batch):
    batch_filtered = list(filter(lambda x: x is not None, batch))
    if not len(batch_filtered):
        return None, None
    return torch.utils.data.dataloader.default_collate(batch_filtered)


def loss_function(prediction, target):
    assert prediction.shape == target.shape

    return torch.sqrt(F.mse_loss(prediction, target, reduction='mean'))


def get_accuracy(outputNN, forecast, target) -> Tuple[float, list[float]]:
    assert outputNN.shape == forecast.shape == target.shape
    mask = torch.logical_or(torch.isnan(target), target == forecast)
    # output_NN[mask] = 0
    # target[mask] = 0
    # forecast[mask] = 0

    magn_NN = torch.sqrt(((outputNN - target) ** 2).nansum(axis=[1, 2, 3, 4]))
    magn_initial = torch.sqrt(((forecast - target) ** 2).nansum(axis=[1, 2, 3, 4]))
    if (magn_initial == 0).sum():
        raise Exception("Found Nans! Should not have happened")

    all_ratios = magn_NN / magn_initial

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


def get_model(args, cfg_neural_network, device):
    model_type = args.model_type
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
        model = OceanCurrentMLP(**cfg_neural_network)

    return model.to(device)


def __create_loss_plot(args, train_losses, test_losses, mean_ratio_train, mean_ratio_test):
    # Plot the loss and the LR

    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(train_losses, color='green', marker='o')
    ax.plot(test_losses, color='red', marker='o')

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss", color='green')
    ax2 = ax.twinx()
    ax2.plot(mean_ratio_train, color='lightgreen')
    ax2.plot(mean_ratio_test, color='orangered')
    ax2.set_ylabel("Mean_ratio", color='blue')
    folder = os.path.abspath(
        args.get("folder_figure", "./") + args["model_type"] + "/" + now.strftime("%d-%m-%Y_%H-%M-%S") + "/")
    filename = f"_loss_and_lr_{len(mean_ratio_train)}.png"
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


def train(args, model, device, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, epoch: int,
          model_error: bool,
          cfg_dataset: dict[str, any]):
    model.train()
    with torch.autograd.set_detect_anomaly(True):
        # for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        total_loss = 0
        all_ratios = []
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, target in tepoch:
                if (data, target) == (None, None):
                    continue
                data, target = data.to(device), target.to(device)
                # todo: adapt in case of window
                axis = cfg_dataset["index_axis_time"]
                shift_input = cfg_dataset.get("shift_window_input", 0)
                # We take the matching input timesteps with the output timesteps
                data_same_time = torch.moveaxis(
                    torch.moveaxis(data, axis, 0)[shift_input:shift_input + target.shape[2]], 0, axis)
                # data_same_time = data.select(axis, 0).unsqueeze(axis)
                optimizer.zero_grad()
                output = model(data)
                if model_error:
                    output = data_same_time - output
                loss = loss_function(output, target)
                total_loss += loss.item()
                # Backprop
                loss.backward()
                # update the weights
                optimizer.step()
                ratio, list_ratios = get_accuracy(output,
                                                  data_same_time,
                                                  target)
                all_ratios += list_ratios
                tepoch.set_postfix(loss=loss.item(), mean_ratio=ratio)
            # wandb.log({'epoch_loss': total_loss / len(train_loader.dataset), })
        total_loss /= len(train_loader)
        __create_histogram(all_ratios, epoch, args, True)
        print(
            f"Training loss: {loss}" + f"% of ratios <= 1: {((np.array(list_ratios) <= 1).sum() / len(list_ratios) * 100):.4f}%, mean ratio{(np.array(list_ratios).mean())}",
        )

        return total_loss, np.array(list_ratios).mean()


def test(args, model, device, test_loader: torch.utils.data.DataLoader, epoch: int,
         model_error: bool, cfg_dataset: dict[str, any]) -> float:
    model.eval()
    test_loss = 0
    initial_loss = 0
    accuracy = 0
    list_ratios = list()

    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            for data, target in tepoch:
                if (data, target) == (None, None):
                    continue
                data, target = data.to(device), target.to(device)
                axis = cfg_dataset["index_axis_time"]
                shift_input = cfg_dataset.get("shift_window_input", 0)
                # We take the matching input timesteps with the output timesteps
                data_same_time = torch.moveaxis(
                    torch.moveaxis(data, axis, 0)[shift_input:shift_input + target.shape[2]], 0, axis)
                output = model(data)
                if model_error:
                    output = data_same_time - output
                loss = loss_function(output, target).item()  # sum the batch losses
                test_loss += loss
                ratio, all_ratios = get_accuracy(output,
                                                 data_same_time,
                                                 target)
                list_ratios += all_ratios
                # TODO: change that
                accuracy += ratio
                initial_loss += loss_function(data_same_time, target).item()
                tepoch.set_postfix(loss=loss, mean_ratio=ratio)

    test_loss /= len(test_loader)
    initial_loss /= len(test_loader)
    accuracy /= len(test_loader)

    __create_histogram(all_ratios, epoch, args, False)
    print(
        f"percentage of ratios <= 1: {((np.array(list_ratios) <= 1).sum() / len(list_ratios) * 100):.4f}% "
        f"Test set: Average loss: {test_loss:.6f}, Without NN: {initial_loss:.6f}"
        f"mean ratio SUM(rmse(NN(FC_xt))/rmse(FC_xt)):({accuracy:.6f}) % increase: {((initial_loss - test_loss) / initial_loss) * 100:.3f}")
    return test_loss, np.array(list_ratios).mean()


def end_training(model, args, train_losses, validation_losses, train_ratios, validation_ratios):
    train_losses = np.array(train_losses)
    validation_losses = np.array(validation_losses)
    train_ratios = np.array(train_ratios)
    validation_ratios = np.array(validation_ratios)
    __create_loss_plot(args, train_losses, validation_losses, train_ratios, validation_ratios)

    print(
        f"Training over. Best validation loss {validation_ratios.min()} at epoch {validation_ratios.argmin()}\n"
        f" with losses train:{train_losses[validation_ratios.argmin()]} test:{validation_losses[validation_ratios.argmin()]}.\n"
        f" List of all the training ratios: {train_losses}")

    if args.save_model:
        torch.save(model.state_dict(), f"{args.model_type}.pt")

    # Log the summary metric using the test set
    # wandb.summary['test_accuracy'] = ...
    wandb.summary['best_validation_ratio'] = validation_ratios.min()
    wandb.finish()


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
    args.setdefault("test_batch_size", 1000)
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
    args.setdefault("model_type", "mlp")
    args.setdefault("max_batches_training_set", -1)
    args.setdefault("max_batches_validation_set", -1)
    return DotDict(args), all_cfgs
    # END ALTERNATIVE


def main():
    wandb.init(project="Seaweed_forecast_improvement", entity="killian2k")  # , name=f"experiment_{}")
    print(f"starting run: {wandb.run.name}")
    os.environ['WANDB_NOTEBOOK_NAME'] = "Seaweed_forecast_improvement"
    # gc.collect()
    parser = argparse.ArgumentParser(description='yaml config file path')
    parser.add_argument('--file-configs', type=str, help='name file config to run (without the extension)')
    config_file = parser.parse_args().file_configs + ".yaml"
    all_cfgs = yaml.load(open(config_file, 'r'),
                         Loader=yaml.FullLoader)
    args, all_cfgs = get_args(all_cfgs)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    torch.manual_seed(args.seed)

    device = device
    print("device:", device)
    if args.silicon:
        device = torch.device("mps")

    cfg_model = all_cfgs.get("model", {})
    cfg_neural_network = cfg_model.get("cfg_neural_network", {}) | {"device": device}
    cfg_dataset = cfg_model.get("cfg_dataset", {})
    cfg_optimizer = args.get("optimizer", {})
    cfg_scheduler = args.get("scheduler", {})
    cfg_data_generation = all_cfgs.get("data_generation", {})
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
    test_kwargs = {'batch_size': args.test_batch_size,
                   'shuffle': cfg_data_generation['parameters_input'].get('shuffle_validation', True)}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    wandb.config.update(args, allow_val_change=True)
    wandb.save(config_file)
    dataset_training = CustomOceanCurrentsFromFiles(folder_training,
                                                    max_items=args.batch_size * args.max_batches_training_set)
    dataset_validation = CustomOceanCurrentsFromFiles(folder_validation,
                                                      max_items=args.batch_size * args.max_batches_validation_set)

    print(f"lengths ds: training:{len(dataset_training)}, validation:{len(dataset_validation)}")

    train_loader = torch.utils.data.DataLoader(dataset_training, collate_fn=collate_fn, **train_kwargs)
    validation_loader = torch.utils.data.DataLoader(dataset_validation, collate_fn=collate_fn, **test_kwargs)

    model = get_model(args, cfg_neural_network, device)
    optimizer = get_optimizer(model, cfg_optimizer.get("name", ""), cfg_optimizer.get("parameters", {}), args.lr)

    # scheduler = schedulers.StepLR(optimizer, step_size=1, gamma=args.gamma)
    scheduler, scheduler_step_takes_argument = get_scheduler(cfg_scheduler, optimizer)
    print(f"optimizer: {optimizer}")

    train_ratios, test_ratios = list(), list()
    train_losses, test_losses = list(), list()
    try:
        for epoch in range(1, args.epochs + 1):
            metrics = {}

            # Training
            print(f"starting Training epoch {epoch}/{args.epochs}.")
            time.sleep(0.2)
            loss, ratio = train(args, model, device, train_loader, optimizer, epoch,
                                model_error, cfg_dataset)
            train_losses.append(loss)
            train_ratios.append(ratio)
            metrics |= {"train_loss": loss, "train_ratio": ratio}

            # Testing
            print(f"starting Testing epoch {epoch}/{args.epochs}.")
            time.sleep(0.2)
            loss, ratio = test(args, model, device, validation_loader, epoch,
                               model_error, cfg_dataset)
            test_losses.append(loss)
            test_ratios.append(ratio)
            metrics |= {"test_loss": loss, "test_ratio": ratio, "learning rate": optimizer.param_groups[0]['lr']}
            if scheduler is not None:
                if scheduler_step_takes_argument:
                    scheduler.step(loss)
                    print(f"current lr: {optimizer.param_groups[0]['lr']}")
                else:
                    scheduler.step()
            wandb.log(metrics)

            if epoch % 30 == 0:
                __create_loss_plot(args, train_losses, test_losses, train_ratios, test_ratios)
    finally:
        end_training(model, args, train_losses, test_losses, train_ratios, test_ratios)


if __name__ == '__main__':
    main()
