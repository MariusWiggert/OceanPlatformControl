import torch
import torch.nn as nn
from ocean_navigation_simulator.generative_error_model.GAN.utils import get_norm_layer


# class Block(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, norm_layer=nn.BatchNorm2d):
#         super(Block, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 4, stride, 2, bias=False, padding_mode="reflect"),
#             nn.MaxPool2d(2),
#             norm_layer(out_channels),
#             nn.LeakyReLU(0.2)
#         )
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# class Discriminator(nn.Module):
#     def __init__(self, in_channels=2, features=(64, 128, 256, 512), norm="batch"):
#         super().__init__()
#         norm_layer = get_norm_layer(norm_type=norm)
#
#         self.initial = nn.Sequential(
#             # in_channels*2 needed because the discriminator receives fake and real at the same time.
#             nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=1, padding=2, padding_mode="reflect"),
#             nn.LeakyReLU(0.2)
#         )
#         layers = []
#         in_channels = features[0]
#         for feature in features[1:]:
#             layers.append(
#                 Block(in_channels,
#                       feature,
#                       stride=1 if feature == features[-1] else 2,
#                       norm_layer=norm_layer)
#             )
#             in_channels = feature
#         layers.append(
#             nn.Conv2d(
#                 in_channels, 1, kernel_size=4, stride=1, padding=2, padding_mode="reflect"
#             )
#         )
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, x, y):
#         x = torch.cat([x, y], dim=1)
#         x = self.initial(x)
#         return self.model(x)
#
#
# def main():
#     x = torch.randn((1, 2, 256, 256))
#     y = torch.randn_like(x)
#     model = Discriminator()
#     preds = model(x, y)
#     print(preds.shape)


class Block(nn.Module):
    """Does not use MaxPool2d unlike other Block. Since currents cannot be mapped to some
    known range (e.g. positives only), MaxPool would ignore negatives."""

    def __init__(self, in_channels, out_channels, stride=2, norm_layer=nn.BatchNorm2d):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            norm_layer(nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"))
            if norm_layer == nn.utils.spectral_norm
            else nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"),
            nn.Identity(out_channels) if norm_layer == nn.utils.spectral_norm
            else norm_layer(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=2, features=(64, 128, 256, 512), norm="batch"):
        super().__init__()
        norm_layer = get_norm_layer(norm_type=norm)

        self.initial = nn.Sequential(
            # in_channels*2 needed because the discriminator receives fake and real at the same time.
            nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                Block(in_channels,
                      feature,
                      stride=1 if feature == features[-1] else 2,
                      norm_layer=norm_layer)
            )
            in_channels = feature
        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)


def main():
    x = torch.randn((1, 2, 256, 256))
    y = torch.randn_like(x)
    model = Discriminator()
    preds = model(x, y)
    print(preds.shape)


if __name__ == "__main__":
    main()


