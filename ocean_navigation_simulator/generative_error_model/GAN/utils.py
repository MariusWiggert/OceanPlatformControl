import torch
from torch.nn import functional as F
from torch.nn import init


# _____________________________ LOSSES ___________________________________ #


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


# _____________________________ NET CONFIGS __________________________________ #


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('Initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('Initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


# _____________________________ SAVING __________________________________ #


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
