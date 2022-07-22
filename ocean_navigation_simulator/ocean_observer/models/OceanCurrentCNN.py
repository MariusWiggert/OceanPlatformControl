from __future__ import print_function

import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from ocean_navigation_simulator.ocean_observer.models.CustomOceanCurrentsDataset import CustomOceanCurrentsDataset


class OceanCurrentCNN(nn.Module):
    def __init__(self):
        super(OceanCurrentCNN, self).__init__()
        self.conv1 = nn.Conv3d(2, 10, 1, 1)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.fc1(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6}")
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
            test_loss += F.mse_loss(output, target, reduction='mean').item()  # sum the batch losses

    test_loss /= len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f})\n")


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
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
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

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

    with open(f'scenarios/neural_networks/{args.yaml_file_datasets}.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset_training = CustomOceanCurrentsDataset(config["training"],
                                                  datetime.datetime(2022, 4, 1, tzinfo=datetime.timezone.utc),
                                                  datetime.datetime(2022, 5, 1, tzinfo=datetime.timezone.utc),
                                                  (36, 36, 120), (48, 48, 12), transform, transform)

    print("done")
    # dataset_testing = CustomOceanCurrentsDataset(config["testing"])
    # dataset2 = datasets.MNIST('../data', train=False,
    #                           transform=transform)
    # train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    # test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    #
    # model = OceanCurrentCNN().to(device)
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    #
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    # for epoch in range(1, args.epochs + 1):
    #     train(args, model, device, train_loader, optimizer, epoch)
    #     test(model, device, test_loader)
    #     scheduler.step()
    #
    # if args.save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
