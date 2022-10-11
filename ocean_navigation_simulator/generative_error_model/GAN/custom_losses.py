import torch


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