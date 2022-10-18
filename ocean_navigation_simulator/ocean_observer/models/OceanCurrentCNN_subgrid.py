from __future__ import print_function

import torch.nn as nn


class OceanCurrentCNNSubgrid(nn.Module):
    def __init__(self):
        super(OceanCurrentCNNSubgrid, self).__init__()
        self.first_layers = nn.Sequential(
            nn.Conv3d(
                in_channels=2, out_channels=32, kernel_size=(24, 4, 4), stride=(4, 2, 2)
            ),  # , stride=(3,3,12))
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2)),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(6, 4, 4), stride=(1, 2, 2)),
            nn.ReLU(),
            # nn.MaxPool3d(kernel_size=(3, 3, 3)),
        )
        input_size = 32 * 7 * 5 * 5
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(),
            nn.Linear(input_size // 4, input_size // 8),
            nn.ReLU(),
        )
        # Upsampling
        self.linear_up_sampling = nn.Sequential(
            nn.Linear(input_size // 8, input_size // 32),
            nn.Linear(input_size // 32, input_size // 32 * 2),
        )

    def forward(self, x):
        # Dims input: [Batch_size, 2 (= dimensions currents), time, lat, lon]
        for layer in self.first_layers:
            x = layer(x)
            print(x.shape, layer)
        for layer in self.linear_layers:
            x = layer(x)
            print(x.shape, layer)
        for layer in self.linear_up_sampling:
            x = layer(x)
            print(x.shape, layer)

        return x
