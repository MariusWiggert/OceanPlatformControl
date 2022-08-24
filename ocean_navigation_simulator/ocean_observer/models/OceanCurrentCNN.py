from __future__ import print_function

import torch
import torch.nn as nn


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
            nn.MaxPool3d(kernel_size=(3, 3, 3)))

        self.linear_layers = nn.Sequential(nn.Flatten(),
                                           nn.Linear(88704, 2000),
                                           nn.ReLU(),
                                           nn.Linear(2000, 1000),
                                           nn.ReLU(),
                                           nn.Linear(1000, 500),
                                           nn.ReLU(),
                                           nn.Linear(500, 250),
                                           nn.ReLU())

        # Upsampling
        self.linear_up_sampling = nn.Linear(250, 88704 // 32)

        self.up_sampling_1 = nn.Sequential(nn.Upsample((35, 42, 55)),  # , scale_factor=(3, 3, 3)),
                                           nn.Conv3d(1, 32, 3, 1, padding='same'),
                                           nn.ReLU())
        # concatenate
        self.up_sampling_2 = nn.Sequential(nn.Conv3d(64, 32, 5, 1),
                                           nn.ReLU(),
                                           nn.Conv3d(32, 32, 5, 1),
                                           nn.ReLU(),

                                           nn.Upsample((118, 140, 236)),  # , scale_factor=(3, 3, 3)),
                                           nn.Conv3d(32, 32, 3, 1, padding='same'),
                                           nn.ReLU())
        # Concatenate
        self.up_sampling_3 = nn.Sequential(nn.Conv3d(32, 32, 5, 1),
                                           nn.ReLU(),
                                           nn.Conv3d(32, 32, 5, 1),
                                           nn.ReLU(),
                                           nn.Conv3d(32, 2, 5, 1)
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
