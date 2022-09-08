from __future__ import print_function

from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch import autograd


class OceanCurrentRNN(nn.Module):
    DIMENSION_TIME_LSTM = 1

    def __init__(self, input_shape, hidden_size, output_shape, type_cell: str, size_layers_after_rnn: List[int] = [],
                 num_layers: int = 1, bidirectional: bool = False, dropout: float = 0,
                 non_linearity: Optional[str] = "tanh", init_weights_function: str = 'randn', index_time_dimension=2,
                 device="cpu"):
        # input shape [batch_size, #currents_dims(2), time, lon, lat]
        super(OceanCurrentRNN, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.index_time_dimension = index_time_dimension
        self.INDEX_BATCH_DIMENSION = 0
        self.device = device
        if init_weights_function == 'randn':
            self.init_weights_function = torch.randn
        elif init_weights_function == 'zeros':
            self.init_weights_function = torch.zeros
        else:
            raise Exception("Unsupported init weights function")
        self.flatten_layer = nn.Flatten(start_dim=2)
        self.type_cell = type_cell.lower()
        self.input_shape = np.array(input_shape)
        self.output_shape = np.array(output_shape)
        input_size = self.input_shape.prod()
        output_size = self.output_shape.prod()
        if self.type_cell == "lstm":
            self.rnn_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                     batch_first=True,
                                     bidirectional=bidirectional, dropout=dropout)
        elif self.type_cell == "rnn":
            self.rnn_layer = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                    batch_first=True,
                                    bidirectional=bidirectional, dropout=dropout, nonlinearity=non_linearity)
        elif self.type_cell == "gru":
            self.rnn_layer = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                    batch_first=True,
                                    bidirectional=bidirectional, dropout=dropout)

        layers_after_RNN = list()
        # layers_after_RNN.append(
        # nn.Conv1d(2 if self.bidirectional else 1, 1, padding=0, kernel_size=hidden_size, stride=hidden_size,
        #          dilation=False))

        # Linear case
        self.final_size = output_size
        layers_after_RNN.append(nn.Flatten())
        layers_after_RNN.append(nn.Linear(hidden_size * (2 if self.bidirectional else 1), self.final_size))
        # Not supported yet
        # layers_after_RNN.append(nn.ReLU())
        # size_layers_after_rnn = [hidden_size] + size_layers_after_rnn
        # for i in range(len(size_layers_after_rnn) - 1):
        #     layers_after_RNN.append(
        #         nn.Linear(in_features=size_layers_after_rnn[i], out_features=size_layers_after_rnn[i + 1]))
        #     layers_after_RNN.append(nn.ReLU())
        #
        # layers_after_RNN.append(nn.Linear(in_features=size_layers_after_rnn[-1], out_features=output_size))
        #
        self.layers_after_RNN = nn.Sequential(*layers_after_RNN)
        print(f"model:\n{self.flatten_layer}\n{self.rnn_layer}\n{self.layers_after_RNN}\n\n")

    def init_hidden(self, batch_size):
        shape = (2 if self.bidirectional else 1) * self.num_layers, batch_size, self.hidden_size
        return autograd.Variable(self.init_weights_function(shape, device=self.device)), autograd.Variable(
            torch.randn(shape, device=self.device))

    def forward(self, x):
        # Dims input: [Batch_size, 2 (= dimensions currents), time, lat, lon]
        batch_size, sequence_length = x.shape[self.INDEX_BATCH_DIMENSION], x.shape[self.index_time_dimension]
        # x becomes shape [batch time, #currents_dim(2),lon,lat]
        x = torch.moveaxis(x, self.index_time_dimension, self.DIMENSION_TIME_LSTM)
        h_0, c_0 = self.init_hidden(batch_size)
        if self.type_cell == "lstm":
            hidden_0 = h_0, c_0
        else:
            hidden_0 = h_0
        x = self.flatten_layer(x)
        output, hidden = self.rnn_layer(x, hidden_0)
        # if self.type_cell == "lstm":
        #     hn, cn = hidden
        # else:
        #     hn = hidden

        # Only consider the last layer of h
        x = output
        x = x.reshape((batch_size * sequence_length, (2 if self.bidirectional else 1), self.hidden_size))

        for l in self.layers_after_RNN:
            x = l(x)

        # Take only the first part
        # Todo: change that
        if self.bidirectional:
            None

        return torch.moveaxis(x.reshape((batch_size, sequence_length, *self.output_shape)), self.DIMENSION_TIME_LSTM,
                              self.index_time_dimension)
