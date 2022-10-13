import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import yaml


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
            self, in_channels=6, out_channels=3, features=(64, 128, 256, 512), dropout=False,
    ):
        super(UNet, self).__init__()
        self.dropout = dropout
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.dropout = nn.Dropout(0.25)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            if self.dropout:
                x = self.dropout(x)

        x = self.bottleneck(x)
        # reverse order or list
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        x = self.final_conv(x)
        # x = self.final_activation(x)
        return x


def test():
    config_file = "../../../scenarios/generative_error_model/config_dl_training.yaml"
    cfgs = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    unet = UNet(in_channels=cfgs["model"]["in_channels"],
                out_channels=cfgs["model"]["out_channels"],
                features=cfgs["model"]["features"])
    test_input = torch.randn((1, 2, 121, 241))
    preds = unet(test_input)
    print(f"Output shape: {preds.shape}.")
    assert preds.shape == test_input.shape, "output shape differs from input shape"


if __name__ == "__main__":
    test()
