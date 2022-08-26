from __future__ import print_function

import argparse
import datetime
import json
import os

import torch
import yaml
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR

from ocean_navigation_simulator.ocean_observer.models.CustomOceanCurrentsDataset import \
    CustomOceanCurrentsDatasetSubgrid
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentCNN_subgrid import OceanCurrentCNNSubgrid
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentsMLP import OceanCurrentMLP


def collate_fn(batch):
    batch_filtered = list(filter(lambda x: x is not None, batch))
    if not len(batch_filtered):
        return None, None
    return torch.utils.data.dataloader.default_collate(batch_filtered)


def loss_function(output, target, mask=None):
    if mask is not None:
        reduction = 'sum'
        division = (mask == False).sum()
    else:
        reduction = 'mean'
        division = 1

    return torch.sqrt(F.mse_loss(output, target, reduction=reduction) / division)


def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    initial_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if (data, target) == (None, None):
                print(f"batch {batch_idx} empty. skipped!")
                continue
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_function(output, target).item()  # sum the batch losses
            # todo: adapt 0 in case where window is around xt instead of just after
            index_xt = 0
            initial_loss += loss_function(data[:, :, [index_xt]], target).item()
            if batch_idx % args.log_interval == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(test_loader.dataset)} ({100. * batch_idx / len(test_loader):.0f}%)]")
                if args.dry_run:
                    break

    test_loss /= len(test_loader.dataset)
    initial_loss /= len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Without NN: {initial_loss:.4f}, Accuracy: {test_loss}/{initial_loss} ({100. * test_loss / initial_loss:.2f})\n")


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

    transform = None
    # transform = transforms.Compose([
    #    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
    # ])

    cfgs = json.load(open(os.getcwd() + "/config/" + args.model_type + ".json", 'r'))
    cfg_model = cfgs.get("cfg_model", {})
    cfg_dataset = cfgs.get("cfg_dataset", {})

    # Load the training and testing files
    with open(f'scenarios/neural_networks/{args.yaml_file_datasets}.yaml') as f:
        config_datasets = yaml.load(f, Loader=yaml.FullLoader)

    # dataset_training = CustomOceanCurrentsDataset(config["training"],
    #                                               datetime.datetime(2022, 4, 1, tzinfo=datetime.timezone.utc),
    #                                               datetime.datetime(2022, 5, 1, tzinfo=datetime.timezone.utc),
    #                                               (36, 36, datetime.timedelta(days=5)),
    #                                               (48, 48, datetime.timedelta(hours=12)), transform, transform)
    # dataset_validation = CustomOceanCurrentsDataset(config["validation"],
    #                                                 datetime.datetime(2022, 4, 1, tzinfo=datetime.timezone.utc),
    #                                                 datetime.datetime(2022, 5, 1, tzinfo=datetime.timezone.utc),
    #                                                 (36, 36, datetime.timedelta(days=5)),
    #                                                 (48, 48, datetime.timedelta(hours=12)), transform, transform)

    start_training = datetime.datetime(2022, 4, 1, 12, 30, 1, tzinfo=datetime.timezone.utc)
    duration_training = datetime.timedelta(days=28)
    duration_validation = datetime.timedelta(days=14)
    # input_tile_dims = (36, 36, datetime.timedelta(days=5))
    # output_tile_dims = (48, 48, datetime.timedelta(hours=12))
    input_tile_dims = (24, 24, datetime.timedelta(hours=5))
    output_tile_dims = (24, 24, datetime.timedelta(hours=1))
    dataset_training = CustomOceanCurrentsDatasetSubgrid(config_datasets["training"], start_training,
                                                         start_training + duration_training,
                                                         input_tile_dims, output_tile_dims, cfg_dataset, transform,
                                                         transform, dtype=dtype, )
    dataset_validation = CustomOceanCurrentsDatasetSubgrid(config_datasets["validation"],
                                                           start_training + duration_training,
                                                           start_training + duration_training + duration_validation,
                                                           input_tile_dims, output_tile_dims, cfg_dataset, transform,
                                                           transform, dtype=dtype, )

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

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        print(f"starting Training epoch {epoch}/{args.epochs + 1}.")

        train(args, model, device, train_loader, optimizer, epoch)
        print(f"starting Testing epoch {epoch}/{args.epochs + 1}.")
        test(args, model, device, validation_loader, epoch)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), f"{args.model_type}.pt")


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        if (data, target) == (None, None):
            print(f"batch {batch_idx} empty. skipped!")
            continue
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        mask = torch.isnan(target)
        output[mask] = 0
        target[mask] = 0
        loss = loss_function(output, target, mask)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6}")
            if args.dry_run:
                break


if __name__ == '__main__':
    main()
