import numpy as np
import tensorflow_probability as tfp
from matplotlib import gridspec, cm, pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

tfk = tfp.math.psd_kernels

n_samples = 1000
n_lines = 4
X = np.linspace(-10.0, 10.0, n_samples).reshape(-1, 1)


def plot_kernel(X, y, sigma, description, fig, subplot_spec, xlim,
                scatter=False, rotate_x_labels=False):
    """Plot kernel matrix and samples."""
    grid_spec = gridspec.GridSpecFromSubplotSpec(
        1, 2, width_ratios=[2, 1], height_ratios=[1],
        wspace=0.18, hspace=0.0,
        subplot_spec=subplot_spec)
    ax1 = fig.add_subplot(grid_spec[0])
    ax2 = fig.add_subplot(grid_spec[1])
    # Plot samples
    if scatter:
        for i in range(y.shape[1]):
            ax1.scatter(X, y[:, i], alpha=0.8, s=3)
    else:
        for i in range(y.shape[1]):
            ax1.plot(X, y[:, i], alpha=0.8)
    ax1.set_ylabel('$y$', fontsize=13, labelpad=0)
    ax1.set_xlabel('$x$', fontsize=13, labelpad=0)
    ax1.set_xlim(xlim)
    if rotate_x_labels:
        for l in ax1.get_xticklabels():
            l.set_rotation(30)
    ax1.set_title(f'Samples from {description}')
    # Plot covariance matrix
    im = ax2.imshow(sigma, cmap=cm.YlGnBu)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.02)
    cbar = plt.colorbar(im, ax=ax2, cax=cax)
    cbar.ax.set_ylabel('$K(X,X)$', fontsize=8)
    ax2.set_title(f'Covariance matrix\n{description}')
    ax2.set_xlabel('X', fontsize=10, labelpad=0)
    ax2.set_ylabel('X', fontsize=10, labelpad=0)
    # Show 5 custom ticks on x an y axis of covariance plot
    nb_ticks = 5
    ticks = list(range(xlim[0], xlim[1] + 1))
    ticks_idx = np.rint(np.linspace(
        1, len(ticks), num=min(nb_ticks, len(ticks))) - 1).astype(int)
    ticks = list(np.array(ticks)[ticks_idx])
    ax2.set_xticks(np.linspace(0, len(X), len(ticks)))
    ax2.set_yticks(np.linspace(0, len(X), len(ticks)))
    ax2.set_xticklabels(ticks)
    ax2.set_yticklabels(ticks)
    if rotate_x_labels:
        for l in ax2.get_xticklabels():
            l.set_rotation(30)
    ax2.grid(False)


# Compute the kernel
sigma = tfk.ExponentiatedQuadratic(amplitude=1., length_scale=1.).matrix(X, X).numpy()
y = np.random.multivariate_normal(mean=np.zeros(n_samples), cov=sigma,
                                  size=n_lines).T

fig = plt.figure(figsize=(7, 10))
gs = gridspec.GridSpec(
    4, 1, figure=fig, wspace=0.2, hspace=0.4)
plot_kernel(
    X, y, sigma, '$\\ell = 1$, $\\sigma = 1$',
    fig, gs[0], (min(X), max(X)))
