import torch
import torch.nn as nn
from ocean_navigation_simulator.generative_error_model.GAN.utils import get_norm_layer


class Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        down=True,
        act="relu",
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        dropout_val=0.5,
    ):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(dropout_val)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        features=64,
        norm="batch",
        dropout_all=False,
        dropout=True,
        dropout_val=0.5,
        latent_size=0,
    ):
        """
        Parameters:
            dropout_all - set dropout on all layers
            dropout - dropout like in pix2pix, first 3 layers of decoder
        """
        super().__init__()
        norm_layer = get_norm_layer(norm_type=norm)

        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(
            features,
            features * 2,
            down=True,
            act="leaky",
            norm_layer=norm_layer,
            use_dropout=dropout_all,
            dropout_val=dropout_val,
        )
        self.down2 = Block(
            features * 2,
            features * 4,
            down=True,
            act="leaky",
            norm_layer=norm_layer,
            use_dropout=dropout_all,
            dropout_val=dropout_val,
        )
        self.down3 = Block(
            features * 4,
            features * 8,
            down=True,
            act="leaky",
            norm_layer=norm_layer,
            use_dropout=dropout_all,
            dropout_val=dropout_val,
        )
        self.down4 = Block(
            features * 8,
            features * 8,
            down=True,
            act="leaky",
            norm_layer=norm_layer,
            use_dropout=dropout_all,
            dropout_val=dropout_val,
        )
        self.down5 = Block(
            features * 8,
            features * 8,
            down=True,
            act="leaky",
            norm_layer=norm_layer,
            use_dropout=dropout_all,
            dropout_val=dropout_val,
        )
        self.down6 = Block(
            features * 8,
            features * 8,
            down=True,
            act="leaky",
            norm_layer=norm_layer,
            use_dropout=dropout_all,
            dropout_val=dropout_val,
        )

        self.bottleneck = nn.Sequential(nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU())

        features_new = features + int(latent_size / 8)
        if dropout:
            self.up1 = Block(
                features * 8,
                features * 8,
                down=False,
                act="relu",
                norm_layer=norm_layer,
                use_dropout=dropout,
                dropout_val=dropout_val,
            )
            self.up2 = Block(
                features * 8 * 2,
                features * 8,
                down=False,
                act="relu",
                norm_layer=norm_layer,
                use_dropout=dropout,
                dropout_val=dropout_val,
            )
            self.up3 = Block(
                features * 8 * 2,
                features * 8,
                down=False,
                act="relu",
                norm_layer=norm_layer,
                use_dropout=dropout,
                dropout_val=dropout_val,
            )
        else:
            self.up1 = Block(
                features_new * 8,
                features * 8,
                down=False,
                act="relu",
                norm_layer=norm_layer,
                use_dropout=dropout_all,
                dropout_val=dropout_val,
            )
            self.up2 = Block(
                features * 8 * 2,
                features * 8,
                down=False,
                act="relu",
                norm_layer=norm_layer,
                use_dropout=dropout_all,
                dropout_val=dropout_val,
            )
            self.up3 = Block(
                features * 8 * 2,
                features * 8,
                down=False,
                act="relu",
                norm_layer=norm_layer,
                use_dropout=dropout_all,
                dropout_val=dropout_val,
            )

        self.up4 = Block(
            features * 8 * 2,
            features * 8,
            down=False,
            act="relu",
            norm_layer=norm_layer,
            use_dropout=dropout_all,
            dropout_val=dropout_val,
        )
        self.up5 = Block(
            features * 8 * 2,
            features * 4,
            down=False,
            act="relu",
            norm_layer=norm_layer,
            use_dropout=dropout_all,
            dropout_val=dropout_val,
        )
        self.up6 = Block(
            features * 4 * 2,
            features * 2,
            down=False,
            act="relu",
            norm_layer=norm_layer,
            use_dropout=dropout_all,
            dropout_val=dropout_val,
        )
        self.up7 = Block(
            features * 2 * 2,
            features,
            down=False,
            act="relu",
            norm_layer=norm_layer,
            use_dropout=dropout_all,
            dropout_val=dropout_val,
        )
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            # nn.Tanh(),
        )

    def forward(self, x, latent=None):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        if latent is not None:
            up1 = self.up1(torch.cat([bottleneck, latent], 1))
        else:
            up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))


class GeneratorSimplified(nn.Module):
    """Keeping all the arguments the same to be compatible with existing train code."""

    def __init__(
        self,
        out_channels=3,
        features=64,
        norm="batch",
        dropout_all=False,
        dropout=True,
        dropout_val=0.5,
        latent_size=0,
    ):

        super().__init__()
        norm_layer = get_norm_layer(norm_type=norm)

        self.up1 = Block(
            latent_size,
            features * 8,
            down=False,
            act="relu",
            norm_layer=norm_layer,
            use_dropout=dropout_all,
            dropout_val=dropout_val,
        )
        self.up2 = Block(
            features * 8,
            features * 8,
            down=False,
            act="relu",
            norm_layer=norm_layer,
            use_dropout=dropout_all,
            dropout_val=dropout_val,
        )
        self.up3 = Block(
            features * 8,
            features * 8,
            down=False,
            act="relu",
            norm_layer=norm_layer,
            use_dropout=dropout_all,
            dropout_val=dropout_val,
        )
        self.up4 = Block(
            features * 8,
            features * 8,
            down=False,
            act="relu",
            norm_layer=norm_layer,
            use_dropout=dropout_all,
            dropout_val=dropout_val,
        )
        self.up5 = Block(
            features * 8,
            features * 4,
            down=False,
            act="relu",
            norm_layer=norm_layer,
            use_dropout=dropout_all,
            dropout_val=dropout_val,
        )
        self.up6 = Block(
            features * 4,
            features * 2,
            down=False,
            act="relu",
            norm_layer=norm_layer,
            use_dropout=dropout_all,
            dropout_val=dropout_val,
        )
        self.up7 = Block(
            features * 2,
            features,
            down=False,
            act="relu",
            norm_layer=norm_layer,
            use_dropout=dropout_all,
            dropout_val=dropout_val,
        )
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features, out_channels, kernel_size=4, stride=2, padding=1),
            # nn.Tanh(),
        )

    def forward(self, x, latent=None):
        up1 = self.up1(latent)
        up2 = self.up2(up1)
        up3 = self.up3(up2)
        up4 = self.up4(up3)
        up5 = self.up5(up4)
        up6 = self.up6(up5)
        up7 = self.up7(up6)
        return self.final_up(up7)


def test():
    x = torch.randn((1, 2, 256, 256))
    latent_size = 512
    model = Generator(
        in_channels=2, out_channels=2, features=64, dropout=False, latent_size=latent_size
    )
    latent = torch.rand((1, latent_size, 1, 1))
    preds = model(x, latent)
    print(preds.shape)


if __name__ == "__main__":
    test()
