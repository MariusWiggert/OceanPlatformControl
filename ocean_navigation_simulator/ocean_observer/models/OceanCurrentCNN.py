from __future__ import print_function

import functools
from typing import List

import torch.nn as nn


class OceanCurrentCNNSubgrid(nn.Module):

    def __init__(self, ch_sz: List[int], device: str = 'cpu', init_weights_value: float = 0.01, activation="relu",
                 downsizing_method="conv", dropout_encoder=0, dropout_decoder=0, dropout_bottom=0,
                 final_number_channels=2, initial_channels: List = None, output_paddings=[(0, 1, 1), (0, 1, 1), 1],
                 instance_norm=False, print_dims=False):
        super(OceanCurrentCNNSubgrid, self).__init__()
        self.init_weights_value = init_weights_value
        self.activation = activation
        self.downsizing_method = downsizing_method
        self.initial_channels = initial_channels
        self.print_dims = print_dims
        # Encoder
        # ch_sz = [2, 16, 32, 64, 92]
        self.bloc_left_1 = nn.Sequential(self.__get_same_dims_bloc(ch_sz[0], ch_sz[1]),
                                         nn.Dropout3d(dropout_encoder),
                                         self.__get_same_dims_bloc(ch_sz[1], ch_sz[1],
                                                                   include_instance_norm=instance_norm),
                                         nn.Dropout3d(dropout_encoder), )
        self.bloc_left_2 = nn.Sequential(self.__get_downsizing_bloc(ch_sz[1], ch_sz[1]),
                                         self.__get_same_dims_bloc(ch_sz[1], ch_sz[2],
                                                                   include_instance_norm=instance_norm),
                                         nn.Dropout3d(dropout_encoder), )
        self.bloc_left_3 = nn.Sequential(self.__get_downsizing_bloc(ch_sz[2], ch_sz[2]),
                                         self.__get_same_dims_bloc(ch_sz[2], ch_sz[3],
                                                                   include_instance_norm=instance_norm),
                                         nn.Dropout3d(dropout_encoder), )

        # If 4 levels
        self.bloc_bottom = nn.Sequential(self.__get_downsizing_bloc(ch_sz[3], ch_sz[3]),
                                         self.__get_same_dims_bloc(ch_sz[3], ch_sz[3],
                                                                   include_instance_norm=instance_norm),
                                         nn.Dropout3d(dropout_bottom), )

        self.bloc_right_3 = nn.Sequential(
            self.__get_upsizing_bloc(ch_sz[3], ch_sz[3], output_padding=output_paddings[0]),
            self.__get_same_dims_bloc(ch_sz[3], ch_sz[3], include_instance_norm=instance_norm),
            nn.Dropout3d(dropout_decoder),
            self.__get_same_dims_bloc(ch_sz[3], ch_sz[3], include_instance_norm=instance_norm),
            nn.Dropout3d(dropout_decoder))
        self.bloc_right_2 = nn.Sequential(
            self.__get_upsizing_bloc(ch_sz[3], ch_sz[2], output_padding=output_paddings[1]),
            self.__get_same_dims_bloc(ch_sz[2], ch_sz[2], include_instance_norm=instance_norm),
            nn.Dropout3d(dropout_decoder),
            self.__get_same_dims_bloc(ch_sz[2], ch_sz[2], include_instance_norm=True),
            nn.Dropout3d(dropout_decoder))
        self.bloc_right_1 = nn.Sequential(
            self.__get_upsizing_bloc(ch_sz[2], ch_sz[1], output_padding=output_paddings[2]),
            self.__get_same_dims_bloc(ch_sz[1], ch_sz[1], include_instance_norm=instance_norm),
            nn.Dropout3d(dropout_decoder),
            self.__get_same_dims_bloc(ch_sz[1], ch_sz[1], include_instance_norm=instance_norm),
            nn.Dropout3d(dropout_decoder))
        self.bloc_final = self.__get_final_bloc(ch_sz[1], final_number_channels)

        all_blocs = [self.bloc_left_1, self.bloc_left_2, self.bloc_left_3, self.bloc_bottom, self.bloc_right_3,
                     self.bloc_right_2, self.bloc_right_1, self.bloc_final]
        [self.__init_weights(bloc) for bloc in all_blocs]

        print("model: ", all_blocs)

    def forward(self, x):
        # Dims input: [Batch_size, #channels, time, lat, lon]
        if self.initial_channels is not None:
            x = x[:, self.initial_channels]
        if self.print_dims:
            print("x", x.shape)
        x1l = self.bloc_left_1(x)
        if self.print_dims:
            print("x1l", x1l.shape, type(x1l), self.bloc_left_2)
        x2l = self.bloc_left_2(x1l)
        if self.print_dims:
            print("x2l", x2l.shape)
        x3l = self.bloc_left_3(x2l)
        if self.print_dims:
            print("x3l", x3l.shape)
        # if 4 levels
        x_bottom = self.bloc_bottom(x3l)
        if self.print_dims:
            print("x_bottom", x_bottom.shape)
        x3r = self.__upsize_and_add_residual(self.bloc_right_3, x_bottom, x3l)
        if self.print_dims:
            print("x3r", x3r.shape)
        x2r = self.__upsize_and_add_residual(self.bloc_right_2, x3r, x2l)
        if self.print_dims:
            print("x2r", x2r.shape)
        x1r = self.__upsize_and_add_residual(self.bloc_right_1, x2r, x1l)
        if self.print_dims:
            print("x1r", x1r.shape)
        return self.bloc_final(x1r)

    def __init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(self.init_weights_value)
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_uniform(m.weight)

    def __get_bloc_unet(self, in_channels: int, out_channels: int, kernel_size, stride, padding,
                        include_instance_norm: bool = False):
        layers = [nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, )]
        nn.init.xavier_uniform(layers[0].weight)
        if include_instance_norm:
            layers.append(nn.InstanceNorm3d(out_channels))
        layers.append(self.__get_activation())
        # nn.MaxPool3d(kernel_size=(3, 3, 3)),
        return nn.Sequential(*layers)

    def __get_activation(self):
        if self.activation.lower() == "relu":
            return nn.ReLU()
        elif self.activation.lower() == "leakyrelu":
            return nn.LeakyReLU()

    def __get_same_dims_bloc(self, in_channels: int, out_channels: int, include_instance_norm: bool = False):
        return self.__get_bloc_unet(in_channels, out_channels, 3, 1, 'same',
                                    include_instance_norm=include_instance_norm)

    def __get_downsizing_bloc(self, in_channels: int, out_channels: int):
        if self.downsizing_method.lower() == "conv":
            return self.__get_bloc_unet(in_channels, in_channels, 3, 2, 1)
        elif self.downsizing_method.lower() == "maxpool":
            return nn.MaxPool3d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        elif self.downsizing_method.lower() == "avgpool":
            return nn.AvgPool3d(kernel_size=3, stride=2, padding=1, ceil_mode=False)

    def __get_upsizing_bloc(self, in_channels: int, out_channels: int, output_padding=0):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                               output_padding=output_padding),
        )

    def __get_final_bloc(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=1, stride=1)
        )

    def __upsize_and_add_residual(self, upsizing_bloc, x_to_upsize, residual):
        if self.print_dims:
            print("dims residual and upsized:", residual.shape, upsizing_bloc[0](x_to_upsize).shape)
        return upsizing_bloc[1:](residual + upsizing_bloc[0](x_to_upsize))

    def get_norm_layer(norm_type='instance'):
        """Return a normalization layer
        Parameters:
            norm_type (str) -- the name of the normalization layer: batch | instance
        BatchNorm, uses learnable affine parameters and track running statistics (mean/stddev).
        InstanceNorm, does not use learnable affine parameters. It does not track running statistics.
        """
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        else:
            raise NotImplementedError(f"Normalization layer {norm_type} is not found")
        return norm_layer
