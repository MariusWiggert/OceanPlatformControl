import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.autograd import Variable
from torch.nn import init
import numpy as np


def snconv2d(
    in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
):
    return spectral_norm(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
    )


def snconv3d(
    in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
):
    return spectral_norm(
        nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
    )


def snlinear(in_features, out_features, bias=True):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=bias))


def sn_embedding(num_embeddings, embedding_dim):
    return spectral_norm(nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim))


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0, gpu=True):
        super(GRU, self).__init__()

        output_size = input_size
        self._gpu = gpu
        self.hidden_size = hidden_size
        self.hidden = None

        # define layers
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear = snlinear(hidden_size, output_size)
        self.bn = nn.BatchNorm1d(output_size, affine=False)

    def forward(self, inputs, n_frames):
        outputs = []
        for i in range(n_frames):
            self.hidden = self.gru(inputs, self.hidden)
            inputs = self.linear(self.hidden)
            outputs.append(inputs)
        outputs = [self.bn(elm) for elm in outputs]
        outputs = torch.stack(outputs)
        return outputs

    def init_weights(self, init_forget_bias=1):
        for name, params in self.named_parameters():
            if "weight" in name:
                init.xavier_uniform_(params)

            # initialize forget gate bias
            elif "gru.bias_ih" in name:
                b_ir, b_iz, b_in = params.chunk(3, 0)
                init.constant_(b_iz, init_forget_bias)
            elif "gru.bias_hh" in name:
                b_hr, b_hz, b_hn = params.chunk(3, 0)
                init.constant_(b_hz, init_forget_bias)
            else:
                init.constant_(params, 0)

    def init_hidden(self, batch_size):
        if batch_size <= 1:
            assert ValueError("Needs to be 2 or more!")
        self.hidden = Variable(torch.zeros(batch_size, self.hidden_size))
        if self._gpu is True:
            self.hidden = self.hidden.cuda()


class CGRU(nn.Module):
    def __init__(self, latent_size, hidden_size, batch):
        super(CGRU, self).__init__()
        self.latent_size = latent_size
        self.batch = batch
        self.fc = snlinear(latent_size, latent_size)
        self.gru = GRU(latent_size, hidden_size, gpu=False)
        self.gru.init_weights()

    def forward(self, z, n_frames):
        h0 = self.fc(z)
        self.gru.init_hidden(self.batch)
        h1 = self.gru(h0, n_frames).transpose(1, 0)
        return h1.contiguous().view(-1, h1.size(2))


def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def test():
    input_size = 128
    hidden_size = 2048
    batch = 2
    n_frames = 192
    # gru = GRU(input_size, hidden_size, gpu=False)
    # gru.init_weights()
    # gru.init_hidden(batch)
    latent = torch.rand((batch, input_size))
    # latent = torch.normal(torch.zeros(batch, input_size), torch.ones(batch, input_size))
    # outputs = gru(latent, n_frames)

    cgru = CGRU(input_size, hidden_size, batch)
    outputs = cgru(latent, n_frames)

    similarities = []
    for i in range(outputs.shape[0] - 1):
        similarities.append(
            cosine_similarity(outputs[i].detach().numpy(), outputs[i + 1].detach().numpy())
        )
    l2 = [
        np.linalg.norm(x)
        for x in np.array(outputs.detach().numpy()).reshape((n_frames * batch, 128))
    ]

    import matplotlib.pyplot as plt

    plt.plot(list(range(batch * n_frames - 1)), similarities, label="cosine similarity")
    plt.plot(list(range(batch * n_frames)), l2, label="l2 norm")
    plt.legend()
    plt.show()

    print(l2[-30:])
    print()
    print(similarities[-30:])


if __name__ == "__main__":
    test()
