from __future__ import print_function

import torch.nn as nn


# Class of the MLP model which was the baseline
class OceanCurrentMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_layers_dims,
        batch_norm_layers,
        dropout_layers,
        output_shape,
        device="cpu",
    ):
        super(OceanCurrentMLP, self).__init__()

        layers = [nn.Flatten()]
        for i in range(len(output_layers_dims)):
            bn = batch_norm_layers[i]
            drop_out = dropout_layers[i]
            inp = input_dim if i == 0 else output_layers_dims[i - 1]
            out = output_layers_dims[i]
            layers.append(nn.Linear(inp, out))
            if bn == 0:
                layers.append(nn.BatchNorm1d(out))
            if i + 1 < len(output_layers_dims):
                layers.append(nn.ReLU())
            if drop_out:
                layers.append(nn.Dropout(drop_out))
            if bn == 1:
                layers.append(nn.BatchNorm1d(out))
        layers.append(nn.Unflatten(1, output_shape))
        self.layers = nn.Sequential(*layers)
        print("model:", self.layers)

    def forward(self, x):
        # Dims input: [Batch_size, 2 (= dimensions currents), time, lat, lon]
        for layer in self.layers:
            x = layer(x)

        return x
