# You need a (free) WandB.ai account.
# avoid excess logging
import logging
import time

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

import wandb

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

wandb.login();

data = scipy.io.loadmat(
    '/Users/fedosha/polybox/semester4/codebase/assessing-pinns-ocean-modelling/Data/Burgers/burgers_shock.mat')
x = np.tile(data['x'], (data['t'].shape[0], 1))  # TN x 1
t = np.repeat(data['t'], data['x'].shape[0], axis=0)  # TN x 1
X = np.concatenate([x, t], axis=1)  # TN x 2
Y = data['usol'].T.reshape(-1, 1)  # TN x 1

print("x.shape:", data['x'].shape)
print("t.shape:", data['t'].shape)
print("u.shape:", data['usol'].shape)
print("X.shape:", X.shape)
print("Y.shape:", Y.shape)

print("X: %s ± %s" % (X.mean(axis=0), X.std(axis=0)))
print("Y: %s ± %s" % (Y.mean(axis=0), Y.std(axis=0)))
X
Y


class PINN_Burgers(nn.Module):
    def __init__(self, layer_width=20, layer_depth=8,
                 activation_function='tanh', initializer='none'):
        super().__init__()

        input_width = 2
        output_width = 1

        self.lambda1 = 1.0
        self.lambda2 = 0.01 / np.pi

        sizes = [input_width] + [layer_width] * layer_depth + [output_width]
        self.net = nn.Sequential(
            *[self.block(dim_in, dim_out, activation_function)
              for dim_in, dim_out in zip(sizes[:-1], sizes[1:-1])],
            nn.Linear(sizes[-2], sizes[-1])  # output layer is regular linear transformation
        )

        if initializer == 'xavier':
            def init_weights(m):
                if type(m) == nn.Linear:
                    torch.nn.init.xavier_uniform_(m.weight)

            self.net.apply(init_weights)

    def forward(self, x):
        return self.net.forward(x)

    def block(self, dim_in, dim_out, activation_function):
        activation_functions = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU()],
            ['relu', nn.ReLU()],
            ['tanh', nn.Tanh()],
            ['sigmoid', nn.Sigmoid()],
            ['softplus', nn.Softplus()],
            ['softsign', nn.Softsign()],
            ['tanhshrink', nn.Tanhshrink()],
            ['celu', nn.CELU()],
            ['gelu', nn.GELU()],
            ['elu', nn.ELU()],
            ['selu', nn.SELU()],
            ['logsigmoid', nn.LogSigmoid()]
        ])
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            activation_functions[activation_function],
        )

    def f(self, x, t, u):
        u_t = grad(u, t, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        u_x = grad(u, x, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        u_xx = grad(u_x, x, create_graph=True, grad_outputs=torch.ones_like(u_x))[0]

        return u_t + self.lambda1 * u * u_x - self.lambda2 * u_xx

    def loss(self, Xu, Yu, Xf=None):
        losses = []
        losses.append(F.mse_loss(self.forward(Xu), Yu))

        if Xf is not None:
            Xf.requires_grad = True
            print("xf is: ", Xf)
            x = Xf[:, 0]
            t = Xf[:, 1]
            Xf = torch.stack((x, t), 1)
            Y_hat = self.forward(Xf)
            u = Y_hat[:, 0]
            f = self.f(x, t, u)
            losses.append(F.mse_loss(f, torch.zeros_like(f)))
        return losses


# parameters
project = "burgers"
epochs = 10000
N = X.shape[0]

torch.manual_seed(2021)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

rng = np.random.default_rng(2021)
seeds = rng.integers(1000, size=5)

# sweep and train
sweep_config = {
    "project": project,
    "method": "grid",
    "parameters": {
        "alpha": {"values": [0.001, 0.1]},
        "Nu": {"values": [200]},
        "Nf": {"values": [12800]},
        "seed": {"values": [0, 1, 2, 3, 4]},
        "func": {"values": ["tanh", "softplus", "relu", "sigmoid", "logsigmoid", "celu", "gelu", "softsign",
                            "tanhshrink"]},
    },
}
sweep_id = wandb.sweep(sweep_config, project=project)

# train
models = []


def model_train():
    run = wandb.init()
    config = wandb.config
    rng = np.random.default_rng(seeds[config.seed])
    name = "burgers_A1s_a%g_nu%g_nf%g_s%d_%s" % (config.alpha, config.Nu, config.Nf, config.seed, config.func)

    # data
    Nu = int(config.Nu)
    Xu_idx = rng.choice(X.shape[0], Nu, replace=False)
    Xu = X[Xu_idx, :]
    Yu = Y[Xu_idx, :]

    Nf = int(config.Nf)
    Xf_idx = rng.choice(X.shape[0], Nf, replace=False)
    Xf = X[Xf_idx, :]

    print("Xu.shape:", Xu.shape)
    print("Yu.shape:", Yu.shape)
    print("Xf.shape:", Xf.shape)

    Xu = torch.tensor(Xu, dtype=torch.float, device=device)
    Yu = torch.tensor(Yu, dtype=torch.float, device=device)
    Xf = torch.tensor(Xf, dtype=torch.float, device=device)
    Xval = torch.tensor(X, dtype=torch.float, device=device)
    Yval = torch.tensor(Y, dtype=torch.float, device=device)

    # model
    model = PINN_Burgers(
        layer_width=20,
        layer_depth=8,
        activation_function=config.func,
        initializer="xavier",
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # training
    val_data_losses = np.array([])
    start_time = time.time()
    for epoch in range(epochs):
        if config.alpha != 0.0:
            losses = model.loss(Xu, Yu, Xf)
            train_data_loss = (1.0 - config.alpha) * losses[0]
            phys_loss = config.alpha * losses[1]
            loss = train_data_loss + phys_loss
        else:
            losses = model.loss(Xu, Yu, None)
            train_data_loss = losses[0]
            phys_loss = torch.tensor(0.0)
            loss = train_data_loss
        wandb.log({"Data loss (training)": train_data_loss.detach().item()}, step=epoch)
        wandb.log({"Physics loss": phys_loss.detach().item()}, step=epoch)
        wandb.log({"Loss (training)": loss.item()}, step=epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses = model.loss(Xval, Yval, None)
        val_data_loss = losses[0]
        val_loss = val_data_loss + phys_loss
        wandb.log({"Data loss (validation)": val_data_loss.item()}, step=epoch)
        wandb.log({"Loss (validation)": val_loss.item()}, step=epoch)

        val_data_losses = np.append(val_data_losses, val_data_loss.item())
        if 250 <= len(val_data_losses):
            lowest_val_data_loss = np.min(val_data_losses[-250:])
        else:
            lowest_val_data_loss = np.min(val_data_losses)
        wandb.log({"Data loss lowest (validation)": lowest_val_data_loss}, step=epoch)

        if epoch % 1000 == 0:
            elapsed = time.time() - start_time
            print("Epoch: %d, Loss: %.3e, Time: %.2fs" % (epoch, val_data_loss, elapsed))
            start_time = time.time()

    torch.save(model.state_dict(), name + ".pth")
    models.append(model)


wandb.agent(sweep_id, project=project, function=model_train)

import matplotlib.pyplot as plt

model_names = ['burgers_A1_w1.pth', 'burgers_A1_w0.pth']

fig, ax = plt.subplots(len(model_names), 3, figsize=(18, 5 * len(model_names)), sharex=True,
                       sharey=True, tight_layout=True, facecolor='white', squeeze=False)
fig.suptitle(
    r'Burgers equation: $\frac{\partial u}{\partial t} = -\lambda_1 u \frac{\partial u}{\partial x} + \lambda_2 \frac{\partial^2 u}{\partial t^2}\;$',
    size=24)

for j, model_name in enumerate(model_names):
    model = PINN_Burgers(layer_width=20,
                         layer_depth=8,
                         activation_function='tanh',
                         initializer='xavier',
                         )
    model.load_state_dict(torch.load(model_name))

    print('Model:', model_name)
    print('  Lambda1:', model.lambda1)
    print('  Lambda2:', model.lambda2)
    Y_hat = model.forward(torch.tensor(X, dtype=torch.float)).detach()
    for i, pos in enumerate([0, 50, 99]):
        start = pos * data['x'].shape[0]
        end = start + data['x'].shape[0]

        if j == 0:
            ax[j, i].set_title('Loss data + physics (t=%d)' % pos, size=16)
        elif j == 1:
            ax[j, i].set_title('Loss data (t=%d)' % pos, size=16)
        ax[j, i].plot(Y[start:end, 0], label='Truth')
        ax[j, i].plot(Y_hat[start:end, 0], label='PINN')
        ax[j, i].legend()
