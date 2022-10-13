import torch
from torch.nn import functional as F


def sparse_mse(predictions, target):
    return torch.where(target != 0, (target - predictions)**2/2, 0).sum()


def total_variation(img, tv_weight: int = 1, reduction: str = "sum"):
    """Compute total variation loss."""
    w_variance = (img[:, :, :, :-1] - img[:, :, :, 1:]).abs()
    h_variance = (img[:, :, :-1, :] - img[:, :, 1:, :]).abs()
    assert w_variance.shape != h_variance.shape, "torch tensor passed as reference!"
    if reduction == "sum":
        w_variance = w_variance.sum()
        h_variance = h_variance.sum()
    elif reduction == "mean":
        w_variance = w_variance.mean()
        h_variance = h_variance.mean()

    loss = tv_weight * (h_variance + w_variance)
    return loss


def mass_conservation(tensor: torch.Tensor, scale: str = "locally"):
    """Computes the mass conservation loss.

    This conservation can either be enforces locally - forcing each 2x2 patch to have zero net flow by
    taking the mse of all net flows.
    Or it can be enforced globally over the whole image by summing all net flows of 2x2 patches and
    taking the absolute value."""

    mc_loss = 0
    top_left = tensor[..., :-1, :-1]
    top_right = tensor[..., :-1, 1:]
    bottom_left = tensor[..., 1:, :-1]
    bottom_right = tensor[..., 1:, 1:]
    all_losses = (-top_left[:, [1]] + top_left[:, [0]] - top_right +
                  bottom_left - bottom_right[:, [0]] + bottom_right[:, [1]]).sum(axis=1)

    # set nans to 0
    total_size = all_losses.nelement()
    num_nans = torch.isnan(all_losses).sum().item()
    all_losses[torch.isnan(all_losses)] = 0
    if scale == "locally":
        mc_loss = torch.sqrt(F.mse_loss(all_losses, torch.zeros_like(all_losses), reduction='sum') / (total_size - num_nans))
    elif scale == "globally":
        mc_loss = torch.abs(all_losses.sum())/(total_size - num_nans)
    return mc_loss
