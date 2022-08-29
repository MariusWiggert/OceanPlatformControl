from __future__ import print_function

import argparse
import os
import time

import torch
import yaml
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from ocean_navigation_simulator.ocean_observer.models.CustomOceanCurrentsFromFiles import CustomOceanCurrentsFromFiles
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentCNN_subgrid import OceanCurrentCNNSubgrid
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentsMLP import OceanCurrentMLP


def collate_fn(batch):
    batch_filtered = list(filter(lambda x: x is not None, batch))
    if not len(batch_filtered):
        return None, None
    return torch.utils.data.dataloader.default_collate(batch_filtered)


def loss_function(output, target, add_mask=False):
    assert output.shape == target.shape
    if add_mask:
        mask = torch.isnan(target)
        output[mask] = 0
        target[mask] = 0
        reduction = 'sum'
        non_nan_elements = (mask == False).sum()
    else:
        reduction = 'mean'
        non_nan_elements = 1

    return torch.sqrt(F.mse_loss(output, target, reduction=reduction) / non_nan_elements)


def train(args, model, device, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, epoch: int,
          model_error: bool,
          cfg_dataset: dict[str, any]):
    model.train()
    with torch.autograd.set_detect_anomaly(True):
        # for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, target in tepoch:
                if (data, target) == (None, None):
                    continue
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                if model_error:
                    # todo: adapt in case of window
                    axis = cfg_dataset["index_axis_time"]
                    output = data.select(axis, 0).unsqueeze(axis) - output
                loss = loss_function(output, target, add_mask=True)
                # Backprop
                loss.backward()
                # update the weights
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())


def test(args, model, device, test_loader: torch.utils.data.DataLoader, epoch: int,
         model_error: bool, cfg_dataset: dict[str, any]) -> None:
    model.eval()
    test_loss = 0
    initial_loss = 0
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            for data, target in tepoch:
                if (data, target) == (None, None):
                    continue
                data, target = data.to(device), target.to(device)
                output = model(data)
                if model_error:
                    # todo: adapt in case of window
                    axis = cfg_dataset["index_axis_time"]
                    output = data.select(axis, 0).unsqueeze(axis) - output
                loss = loss_function(output, target, add_mask=True).item()  # sum the batch losses
                test_loss += loss
                # todo: adapt 0 in case where window is around xt instead of just after
                index_xt = 0
                initial_loss += loss_function(data[:, :, [index_xt]], target, add_mask=True).item()
                tepoch.set_postfix(loss=loss, accuracy=(100. * test_loss / initial_loss))

    test_loss /= len(test_loader.dataset)
    initial_loss /= len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.6f}, Without NN: {initial_loss:.6f}, ratio NN_loss/initial_loss:({100. * test_loss / initial_loss:.4f}%)\n")


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch model')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--yaml-file-datasets', type=str, default='',
                        help='filname of the yaml file to use to download the data in the folder scenarios/neural_networks')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--silicon', action='store_true', default=False,
                        help='enable Mac silicon optimization')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--model-type', type=str, default='mlp')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print("device:", device)
    dtype = torch.float32
    if args.silicon:
        dtype = torch.float
        device = torch.device("mps")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    cfgs = yaml.load(open(os.getcwd() + "/config/" + args.model_type + ".yaml", 'r'), Loader=yaml.FullLoader)
    cfg_model = cfgs.get("cfg_model", {})
    cfg_dataset = cfgs.get("cfg_dataset", {})
    model_error = cfgs.get("model_error", True)
    print("The Model will predict the " + ("error" if model_error else "hindcast") + ".")

    # Load the training and testing files
    # with open(f'scenarios/neural_networks/{args.yaml_file_datasets}.yaml') as f:
    #    config_datasets = yaml.load(f, Loader=yaml.FullLoader)

    dataset_training = CustomOceanCurrentsFromFiles("./data_NN/data_exported_2/")
    dataset_validation = CustomOceanCurrentsFromFiles("./data_NN/data_exported_2/")

    train_loader = torch.utils.data.DataLoader(dataset_training, collate_fn=collate_fn, **train_kwargs)
    validation_loader = torch.utils.data.DataLoader(dataset_validation, collate_fn=collate_fn, **test_kwargs)
    model_type = args.model_type
    if model_type == 'mlp':
        model = OceanCurrentMLP(**cfg_model)
    elif model_type == 'cnn':
        model = OceanCurrentCNNSubgrid(**cfg_model)
    else:
        model = OceanCurrentMLP(**cfg_model)

    model = model.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    print(f"optimizer: {optimizer}")

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        print(f"\nstarting Training epoch {epoch}/{args.epochs + 1}.")
        time.sleep(0.2)
        train(args, model, device, train_loader, optimizer, epoch, model_error, cfg_dataset)
        time.sleep(0.2)
        print(f"\nstarting Testing epoch {epoch}/{args.epochs + 1}.")
        test(args, model, device, validation_loader, epoch, model_error, cfg_dataset)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), f"{args.model_type}.pt")


if __name__ == '__main__':
    main()
