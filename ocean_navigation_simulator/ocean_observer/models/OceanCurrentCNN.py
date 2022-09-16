from __future__ import print_function

import torch.nn as nn


class OceanCurrentCNNSubgrid(nn.Module):

    def __init__(self, ch_sz: list[int], device: str = 'cpu', init_weights_value: float = 0.01, activation="relu",
                 downsizing_method="conv", dropout_encoder=0, dropout_decoder=0, dropout_bottom=0,
                 final_number_channels=2):
        super(OceanCurrentCNNSubgrid, self).__init__()
        self.init_weights_value = init_weights_value
        self.activation = activation
        self.downsizing_method = downsizing_method
        # Encoder
        # ch_sz = [2, 16, 32, 64, 92]
        self.bloc_left_1 = nn.Sequential(self.__get_same_dims_bloc(ch_sz[0], ch_sz[1]),
                                         nn.Dropout3d(dropout_encoder),
                                         self.__get_same_dims_bloc(ch_sz[1], ch_sz[1]),
                                         nn.Dropout3d(dropout_encoder), )
        self.bloc_left_2 = nn.Sequential(self.__get_downsizing_bloc(ch_sz[1], ch_sz[1]),
                                         self.__get_same_dims_bloc(ch_sz[1], ch_sz[2]),
                                         nn.Dropout3d(dropout_encoder), )
        self.bloc_left_3 = nn.Sequential(self.__get_downsizing_bloc(ch_sz[2], ch_sz[2]),
                                         self.__get_same_dims_bloc(ch_sz[2], ch_sz[3]),
                                         nn.Dropout3d(dropout_encoder), )

        # If 4 levels
        self.bloc_bottom = nn.Sequential(self.__get_downsizing_bloc(ch_sz[3], ch_sz[3]),
                                         self.__get_same_dims_bloc(ch_sz[3], ch_sz[3]),
                                         nn.Dropout3d(dropout_bottom), )

        self.bloc_right_3 = nn.Sequential(self.__get_upsizing_bloc(ch_sz[3], ch_sz[3], output_padding=(0, 1, 1)),
                                          self.__get_same_dims_bloc(ch_sz[3], ch_sz[3]),
                                          nn.Dropout3d(dropout_decoder),
                                          self.__get_same_dims_bloc(ch_sz[3], ch_sz[3]),
                                          nn.Dropout3d(dropout_decoder))
        self.bloc_right_2 = nn.Sequential(self.__get_upsizing_bloc(ch_sz[3], ch_sz[2], output_padding=(1, 1, 1)),
                                          self.__get_same_dims_bloc(ch_sz[2], ch_sz[2]),
                                          nn.Dropout3d(dropout_decoder),
                                          self.__get_same_dims_bloc(ch_sz[2], ch_sz[2]),
                                          nn.Dropout3d(dropout_decoder))
        self.bloc_right_1 = nn.Sequential(self.__get_upsizing_bloc(ch_sz[2], ch_sz[1], output_padding=1),
                                          self.__get_same_dims_bloc(ch_sz[1], ch_sz[1]),
                                          nn.Dropout3d(dropout_decoder),
                                          self.__get_same_dims_bloc(ch_sz[1], ch_sz[1]),
                                          nn.Dropout3d(dropout_decoder))
        self.bloc_final = self.__get_final_bloc(ch_sz[1], final_number_channels)

        all_blocs = [self.bloc_left_1, self.bloc_left_2, self.bloc_left_3, self.bloc_bottom, self.bloc_right_3,
                     self.bloc_right_2, self.bloc_right_1, self.bloc_final]
        [self.__init_weights(bloc) for bloc in all_blocs]

        print("model: ", all_blocs)

    def forward(self, x):
        # Dims input: [Batch_size, 2 (= dimensions currents), time, lat, lon]
        x1l = self.bloc_left_1(x)
        x2l = self.bloc_left_2(x1l)
        x3l = self.bloc_left_3(x2l)
        # if 4 levels
        x_bottom = self.bloc_bottom(x3l)
        x3r = self.__upsize_and_add_residual(self.bloc_right_3, x_bottom, x3l)
        x2r = self.__upsize_and_add_residual(self.bloc_right_2, x3r, x2l)
        x1r = self.__upsize_and_add_residual(self.bloc_right_1, x2r, x1l)
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

    def __get_same_dims_bloc(self, in_channels: int, out_channels: int):
        return self.__get_bloc_unet(in_channels, out_channels, 3, 1, 'same')

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
        return upsizing_bloc[1:](residual + upsizing_bloc[0](x_to_upsize))
