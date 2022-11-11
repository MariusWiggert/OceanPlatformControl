from __future__ import print_function

from typing import List

import torch
import torch.nn as nn

from ocean_navigation_simulator.ocean_observer.models.OceanCurrentsRNN import (
    OceanCurrentRNN,
)


# Class of the 2D U-net LSTM models.
class OceanCurrentUnetLSTM(nn.Module):
    dim_currents, dim_time = 1, 2

    def __init__(
        self,
        ch_sz: List[int],
        device: str = "cpu",
        init_weights_value: float = 0.01,
        activation="relu",
        downsizing_method="conv",
        lstm_hidden=24,
    ):
        super(OceanCurrentUnetLSTM, self).__init__()
        self.init_weights_value = init_weights_value
        self.activation = activation
        self.downsizing_method = downsizing_method
        print(f"self.downsizing_method: {self.downsizing_method}")
        # Encoder
        # ch_sz = [2, 16, 32, 64, 92]
        self.bloc_left_1 = nn.Sequential(
            self.__get_same_dims_bloc(ch_sz[0], ch_sz[1]),
            self.__get_same_dims_bloc(ch_sz[1], ch_sz[1]),
        )
        self.bloc_left_2 = nn.Sequential(
            self.__get_downsizing_bloc(ch_sz[1], ch_sz[1]),
            self.__get_same_dims_bloc(ch_sz[1], ch_sz[2]),
        )
        self.bloc_left_3 = nn.Sequential(
            self.__get_downsizing_bloc(ch_sz[2], ch_sz[2]),
            self.__get_same_dims_bloc(ch_sz[2], ch_sz[3]),
        )

        # If 4 levels
        self.bloc_bottom = nn.Sequential(
            self.__get_downsizing_bloc(ch_sz[3], ch_sz[3]),
            self.__get_same_dims_bloc(ch_sz[3], ch_sz[3]),
        )

        self.bloc_right_3 = nn.Sequential(
            self.__get_upsizing_bloc(ch_sz[3], ch_sz[3], output_padding=1),
            self.__get_same_dims_bloc(ch_sz[3], ch_sz[3]),
            self.__get_same_dims_bloc(ch_sz[3], ch_sz[3]),
        )
        self.bloc_right_2 = nn.Sequential(
            self.__get_upsizing_bloc(ch_sz[3], ch_sz[2], output_padding=1),
            self.__get_same_dims_bloc(ch_sz[2], ch_sz[2]),
            self.__get_same_dims_bloc(ch_sz[2], ch_sz[2]),
        )
        self.bloc_right_1 = nn.Sequential(
            self.__get_upsizing_bloc(ch_sz[2], ch_sz[1], output_padding=1),
            self.__get_same_dims_bloc(ch_sz[1], ch_sz[1]),
            self.__get_same_dims_bloc(ch_sz[1], ch_sz[1]),
        )
        self.bloc_final = self.__get_final_bloc(ch_sz[1], ch_sz[0])

        all_blocs = [
            self.bloc_left_1,
            self.bloc_left_2,
            self.bloc_left_3,
            self.bloc_bottom,
            self.bloc_right_3,
            self.bloc_right_2,
            self.bloc_right_1,
            self.bloc_final,
        ]
        [self.__init_weights(bloc) for bloc in all_blocs]

        input_size = [2, 24, 24]
        bidirectional = False
        dropout = 0.2
        # self.flatten = nn.Flatten(2)
        # self.lstm = nn.LSTM(input_size, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.lstm = OceanCurrentRNN(
            input_size,
            lstm_hidden,
            [2, 24, 24],
            "lstm",
            num_layers=2,
            bidirectional=bidirectional,
            dropout=dropout,
            index_time_dimension=self.dim_time,
            device=device,
        )
        all_blocs.append(self.lstm)
        print("model: ", all_blocs)

    def forward(self, x):
        # Dims input: [Batch_size, 2 (= dimensions currents), time, lat, lon]
        x_cnn = torch.swapaxes(x, self.dim_currents, self.dim_time)
        # Dims x_cnn: [Batch_size, time,  2 (= dimensions currents), lat, lon]
        batch_size, seq_len, currents_dim, lat, lon = x_cnn.shape
        x_cnn = x_cnn.reshape(-1, currents_dim, lat, lon)
        assert currents_dim == 2

        x1l = self.bloc_left_1(x_cnn)
        x2l = self.bloc_left_2(x1l)
        x3l = self.bloc_left_3(x2l)

        # if 4 levels
        x_bottom = self.bloc_bottom(x3l)
        x3r = self.upsize_and_add_residual(self.bloc_right_3, x_bottom, x3l)
        x2r = self.upsize_and_add_residual(self.bloc_right_2, x3r, x2l)
        x1r = self.upsize_and_add_residual(self.bloc_right_1, x2r, x1l)
        x_cnn = self.bloc_final(x1r)

        x_pre_lstm = torch.swapaxes(
            x_cnn.reshape(batch_size, seq_len, currents_dim, lat, lon),
            self.dim_time,
            self.dim_currents,
        )
        return self.lstm(x_pre_lstm)
        # x_pre_lstm = self.flatten(x_pre_lstm)
        # x_post_lstm = self.lstm(x_pre_lstm)

    def __init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(self.init_weights_value)
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight)

    def __get_bloc_unet(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride,
        padding,
        include_instance_norm: bool = False,
    ):
        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        ]
        nn.init.xavier_uniform(layers[0].weight)
        if include_instance_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(self.__get_activation())
        return nn.Sequential(*layers)

    def __get_activation(self):
        if self.activation.lower() == "relu":
            return nn.ReLU()
        elif self.activation.lower() == "leakyrelu":
            return nn.LeakyReLU()

    def __get_same_dims_bloc(self, in_channels: int, out_channels: int):
        return self.__get_bloc_unet(in_channels, out_channels, 3, 1, "same")

    def __get_downsizing_bloc(self, in_channels: int, out_channels: int):
        if self.downsizing_method.lower() == "conv":
            return self.__get_bloc_unet(in_channels, out_channels, 3, 2, 1)
        elif self.downsizing_method.lower() == "maxpool":
            return nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif self.downsizing_method.lower() == "avgpool":
            return nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def __get_upsizing_bloc(self, in_channels: int, out_channels: int, output_padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=output_padding,
            ),
        )

    def __get_final_bloc(self, in_channels: int, out_channels: int):
        return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=1))

    def upsize_and_add_residual(self, upsizing_bloc, x_to_upsize, residual):
        return upsizing_bloc[1:](residual + upsizing_bloc[0](x_to_upsize))
