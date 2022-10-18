from __future__ import print_function

import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import optim
from torch.optim.lr_scheduler import StepLR

from ocean_navigation_simulator.ocean_observer.models.CustomOceanCurrentsDataset import (
	CustomOceanCurrentsDatasetSubgrid,
)
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentCNN_subgrid import (
	OceanCurrentCNNSubgrid,
)


class OceanCurrentCNN(nn.Module):
    def __init__(self):
        super(OceanCurrentCNN, self).__init__()
        # radius_output = 12
        # margin_input = 6
        # width_input = (radius_output + margin_input) * 2

        # self.conv1 = nn.Conv3d(in_channels=2, out_channels=32, kernel_size=(5, 5, 24))  # , stride=(3,3,12))
        # self.act1 = nn.ReLU()
        # self.pool1 = nn.MaxPool3d(kernel_size=(3, 3, 3))
        #
        # self.conv2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(5, 5, 24))
        # self.act2 = nn.ReLU()
        # self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3))
        #
        # self.flatten = nn.Flatten()
        # self.dense1 = nn.Linear(88704, 2000)
        # self.dense1_act = nn.ReLU()
        # self.dense2 = nn.Linear(2000, 1000)
        # self.dense2_act = nn.ReLU()
        # self.dense3 = nn.Linear(1000, 500)
        # self.dense3_act = nn.ReLU()
        # self.dense4 = nn.Linear(500, 250)
        # self.dense4_act = nn.ReLU()
        self.first_layers = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=32, kernel_size=(5, 5, 24)),  # , stride=(3,3,12))
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 3)),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(5, 5, 24)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 3)),
        )

        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(88704, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
        )

        # Upsampling
        self.linear_up_sampling = nn.Linear(250, 88704 // 32)

        self.up_sampling_1 = nn.Sequential(
            nn.Upsample((35, 42, 55)),  # , scale_factor=(3, 3, 3)),
            nn.Conv3d(1, 32, 3, 1, padding="same"),
            nn.ReLU(),
        )
        # concatenate
        self.up_sampling_2 = nn.Sequential(
            nn.Conv3d(64, 32, 5, 1),
            nn.ReLU(),
            nn.Conv3d(32, 32, 5, 1),
            nn.ReLU(),
            nn.Upsample((118, 140, 236)),  # , scale_factor=(3, 3, 3)),
            nn.Conv3d(32, 32, 3, 1, padding="same"),
            nn.ReLU(),
        )
        # Concatenate
        self.up_sampling_3 = nn.Sequential(
            nn.Conv3d(32, 32, 5, 1),
            nn.ReLU(),
            nn.Conv3d(32, 32, 5, 1),
            nn.ReLU(),
            nn.Conv3d(32, 2, 5, 1),
        )

    def forward(self, x):

        # Dims input: [Batch_size, 2 (= dimensions currents), time, lat, lon]
        print("Forward:")
        inputs_saved = []
        batch_size = x.shape[0]
        for layer in self.first_layers:
            if isinstance(layer, nn.MaxPool3d):
                inputs_saved.append(x)
            x = layer(x)
        before_linear_shape = x.shape
        for layer in self.linear_layers:
            x = layer(x)

        x = self.linear_up_sampling(x)
        x = x.reshape((batch_size, 1, *before_linear_shape[2:]))

        for layer in self.up_sampling_1:
            x = layer(x)
        x = torch.cat((x, inputs_saved[1]), 1)
        for layer in self.up_sampling_2:
            x = layer(x)
        x = torch.cat((x, inputs_saved[0]), -1)
        print("x shape after cat:", x.shape)
        for layer in self.up_sampling_3:
            x = layer(x)
            print(x.shape, layer)
        print(x.shape)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        print("output_train: ", output.shape, "target:", target.shape)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6}"
            )
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            print("output shape:", output.shape)
            test_loss += F.mse_loss(output, target, reduction="mean").item()  # sum the batch losses

    test_loss /= len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f})\n"
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--yaml-file-datasets",
        type=str,
        default="",
        help="filname of the yaml file to use to download the data in the folder scenarios/neural_networks",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False, help="quickly check a single pass"
    )
    parser.add_argument(
        "--silicon", action="store_true", default=False, help="enable Mac silicon optimization"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model", action="store_true", default=False, help="For Saving the current Model"
    )

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    if args.silicon:
        device = torch.device("mps")
    print("device:", device)

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = None
    # transform = transforms.Compose([
    #    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
    # ])

    with open(f"scenarios/neural_networks/{args.yaml_file_datasets}.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

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

    dataset_training = CustomOceanCurrentsDatasetSubgrid(
        config["training"],
        datetime.datetime(2022, 4, 1, 12, 30, 1, tzinfo=datetime.timezone.utc),
        datetime.datetime(2022, 4, 1, 12, 30, 1, tzinfo=datetime.timezone.utc)
        + datetime.timedelta(days=28),
        (36, 36, datetime.timedelta(days=5)),
        (48, 48, datetime.timedelta(hours=12)),
        transform,
        transform,
    )
    dataset_validation = CustomOceanCurrentsDatasetSubgrid(
        config["validation"],
        datetime.datetime(2022, 4, 1, tzinfo=datetime.timezone.utc),
        datetime.datetime(2022, 5, 1, tzinfo=datetime.timezone.utc),
        (36, 36, datetime.timedelta(days=5)),
        (48, 48, datetime.timedelta(hours=12)),
        transform,
        transform,
    )

    train_loader = torch.utils.data.DataLoader(dataset_training, **train_kwargs)
    validation_loader = torch.utils.data.DataLoader(dataset_validation, **test_kwargs)

    model = OceanCurrentCNNSubgrid().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, validation_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "cnn.pt")


if __name__ == "__main__":
    main()
