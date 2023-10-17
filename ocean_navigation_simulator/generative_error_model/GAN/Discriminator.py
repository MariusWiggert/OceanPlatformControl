import torch
import torch.nn as nn
from ocean_navigation_simulator.generative_error_model.GAN.utils import get_norm_layer


class Block(nn.Module):
    """Does not use MaxPool2d unlike other Block. Since currents cannot be mapped to some
    known range (e.g. positives only), MaxPool would ignore negatives."""

    def __init__(self, in_channels, out_channels, stride=2, norm_layer=nn.BatchNorm2d):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            norm_layer(
                nn.Conv2d(
                    in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"
                )
            )
            if norm_layer == nn.utils.spectral_norm
            else nn.Conv2d(
                in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"
            ),
            nn.Identity(out_channels)
            if norm_layer == nn.utils.spectral_norm
            else norm_layer(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels=2,
        features=(64, 128, 256, 512),
        norm="batch",
        patch_disc=True,
        conditional=True,
    ):
        super().__init__()
        norm_layer = get_norm_layer(norm_type=norm)
        self.patch_disc = patch_disc
        self.conditional = conditional

        self.initial = nn.Sequential(
            # in_channels*2 needed because the discriminator receives fake and real at the same time.
            nn.Conv2d(
                in_channels * 2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            )
            if conditional
            else nn.Conv2d(
                in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"
            ),
            nn.LeakyReLU(0.2),
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                Block(
                    in_channels,
                    feature,
                    stride=1 if feature == features[-1] else 2,
                    norm_layer=norm_layer,
                )
            )
            in_channels = feature
        layers.append(
            nn.Conv2d(
                in_channels,
                1 if patch_disc else in_channels * 2,
                kernel_size=4,
                stride=1 if patch_disc else 2,
                padding=1 if patch_disc else 1,
                padding_mode="reflect",
            )
        )
        self.model = nn.Sequential(*layers)

        # add extra layers if not patch disc
        self.final = nn.Sequential(
            norm_layer(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels * 2,
                in_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            norm_layer(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels * 2,
                in_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            norm_layer(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels * 2, 1, kernel_size=3, stride=2, padding=0, padding_mode="reflect"
            ),
        )

    def forward(self, y, x=None):
        """
        Arguments:
            y - real/fake data
            x - FC (data that conditioned on)
        """

        if self.conditional:
            x = torch.cat([x, y], dim=1)
        else:
            x = y
        x = self.initial(x)
        if self.patch_disc:
            return self.model(x)
        else:
            x = self.model(x)
            return self.final(x)


def main():
    x = torch.randn((1, 2, 256, 256))
    y = torch.randn_like(x)
    model = Discriminator(patch_disc=False, conditional=False)
    preds = model(y, x)
    print(preds.shape)


if __name__ == "__main__":
    main()
