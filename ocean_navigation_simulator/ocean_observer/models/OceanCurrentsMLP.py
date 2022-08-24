from __future__ import print_function

import torch.nn as nn


class OceanCurrentMLP(nn.Module):
    def __init__(self, input_dim, output_layers_dims, output_shape):
        super(OceanCurrentMLP, self).__init__()

        layers = [nn.Flatten()]
        for i in range(len(output_layers_dims)):
            inp = input_dim if i == 0 else output_layers_dims[i - 1]
            out = output_layers_dims[i]
            layers.append(nn.Linear(inp, out))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out))
        layers.append(nn.Unflatten(1, output_shape))
        self.layers = nn.Sequential(*layers)
        print("model:", self.layers)

    def forward(self, x):
        # Dims input: [Batch_size, 2 (= dimensions currents), time, lat, lon]
        for l in self.layers:
            x = l(x)

        return x
