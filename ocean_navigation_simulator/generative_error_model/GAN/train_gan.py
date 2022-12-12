from ocean_navigation_simulator.generative_error_model.GAN.utils import l1, mse, sparse_mse, total_variation,\
    mass_conservation, init_weights, save_checkpoint, load_checkpoint
from ocean_navigation_simulator.generative_error_model.generative_model_metrics import rmse, vector_correlation
from ocean_navigation_simulator.generative_error_model.GAN.ssim import ssim
from ocean_navigation_simulator.generative_error_model.GAN.helper_funcs import get_model, get_data, get_test_data,\
    get_optimizer, get_scheduler, initialize, save_input_output_pairs, enable_dropout

import wandb
import os
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from typing import Dict, Tuple, List, Optional
from warnings import warn
from tqdm import tqdm


def predict_fixed_batch(model, dataloader, device) -> dict:
    """Performs prediction on a fixed batch to be able to assess performance qualitatively.
    This batch is visualized and saved on weights and biases."""

    model.eval()
    with torch.no_grad():
        data = next(iter(dataloader))
        samples = data[0]
        ground_truth = data[1]
        samples = samples.to(device)
        model_output = model(samples).cpu().detach()
        samples = samples.cpu().detach()[:, :2]

    # logging final images to weights and biases
    ground_truth = make_grid(ground_truth, 2)
    predictions = make_grid(model_output, 2)
    samples = make_grid(samples, 2)
    ground_truth = wandb.Image(ground_truth, caption="Fixed batch samples")
    predictions = wandb.Image(predictions, caption="Fixed batch predictions")
    samples = wandb.Image(samples, caption="Fixed batch samples")
    model.train()
    return {"fixed_batch_gt": ground_truth, "fixed_batch_predictions": predictions, "fixed_batch_inputs": samples}


def get_metrics(metric_names, ground_truth: torch.Tensor, predictions: torch.Tensor) -> Optional[Dict[str, float]]:
    """Computes all specified metrics over the validation set predictions.
    Computes metrics for each pair, sums the result of all pairs in batch and returns metrics
    normalized by batch size."""

    # convert to numpy
    ground_truth = ground_truth.cpu().detach().numpy()
    predictions = predictions.cpu().detach().numpy()

    # compute metrics
    metrics = {metric_name: 0 for metric_name in metric_names}
    for idx in range(ground_truth.shape[0]):
        if "rmse" in metric_names:
            rmse_val = rmse(ground_truth[idx, :2, :, :].squeeze(),
                            predictions[idx, :2, :, :].squeeze())
            metrics["rmse"] += rmse_val
        if "vector_correlation" in metric_names:
            vec_corr_val = vector_correlation(ground_truth[idx, 0, :, :].squeeze(),
                                              ground_truth[idx, 1, :, :].squeeze(),
                                              predictions[idx, 0, :, :].squeeze(),
                                              predictions[idx, 1, :, :].squeeze())
            metrics["vector_correlation"] += vec_corr_val
        if "ssim" in metric_names:
            ssim_val = ssim(torch.Tensor(ground_truth[idx].squeeze()),
                            torch.Tensor(predictions[idx].squeeze()))
            metrics["ssim"] += ssim_val
    metrics = {metric_name: metric_value / ground_truth.shape[0] for metric_name, metric_value in metrics.items()}
    return metrics


def loss_function(losses: List[str], loss_weightings: List[float], target: Optional[torch.Tensor] = None,
                  target_fake: Optional[torch.Tensor] = None, disc_real: Optional[torch.Tensor] = None,
                  disc_fake: Optional[torch.Tensor] = None):
    """Handles which loss is to be used.
    """
    loss = 0
    loss_map = {
        "mse": mse,
        "l1": l1,
        "sparse_mse": sparse_mse,
        "total_variation": total_variation,
        "mass_conservation": mass_conservation,
        "gan_gen": gan_loss_gen,
        "gan_disc": gan_loss_disc
    }

    if len(losses) != len(loss_weightings):
        raise ValueError("'losses' and 'loss_weightings' are not the same length!")

    for loss_type, weight in zip(losses, loss_weightings):
        if loss_type in ["mse", "l1", "sparse_mse"]:
            loss += weight * loss_map[loss_type](target_fake, target)
        elif loss_type == "gan_gen":
            loss += weight * loss_map[loss_type](disc_fake)
        elif loss_type == "gan_disc":
            loss += weight * loss_map[loss_type](disc_real, disc_fake)
        else:
            loss += weight * loss_map[loss_type](target_fake)

    if loss == 0:
        raise warn("Loss is zero!")
    return loss


def gan_loss_disc(disc_real, disc_fake):
    """
    Parameters:
        disc_real - output of discriminator for real examples
        disc_fake - output of discriminaor for fake examples
    """
    bce_loss = nn.BCEWithLogitsLoss()
    disc_real_loss = bce_loss(disc_real, torch.ones_like(disc_real))
    disc_fake_loss = bce_loss(disc_fake, torch.zeros_like(disc_fake))
    disc_loss = (disc_real_loss + disc_fake_loss) / 2  # division to make disc learn more slowly
    return disc_loss


def gan_loss_gen(disc_fake):
    """
    Parameters:
        disc_fake - output of discriminator for fake examples
        target_fake - output of generator
        target - ground truth data
    """
    bce_loss = nn.BCEWithLogitsLoss()
    gen_loss = bce_loss(disc_fake, torch.ones_like(disc_fake))
    return gen_loss


def train(models: Tuple[nn.Module, nn.Module], optimizers, dataloader, device, cfgs_train):
    gen_loss_sum, disc_loss_sum = 0, 0
    [model.train() for model in models]
    with torch.enable_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Training epoch [{cfgs_train['epoch']}/{cfgs_train['epochs']}]")
            for data, target in tepoch:
                data, target = data.to(device).float(), target.to(device).float()

                # train discriminator
                target_fake = models[0](data)
                # print(f"target nan values: {torch.isnan(target[0, 0]).sum()}")
                # print(f"target num all zero frames: "
                #       f"{[torch.where(target[0, i] != 0, 1, 0).sum()for i in range(target.shape[1])].count(0)}")
                # mask the generator output to match target (buoy data)
                mask = torch.where(target != 0, 1, 0)
                target_fake = torch.mul(target_fake, mask).float()
                # check whether same num of non-zero values in target and target_fake
                assert mask.sum() == torch.where(target_fake != 0, 1, 0).sum()
                # print(target_fake[0, 0])
                # compute real and fake outputs of discriminator
                disc_real = models[1](data, target)
                disc_fake = models[1](data, target_fake)  # need to call detach to remove from comp graph
                disc_loss = loss_function(cfgs_train["loss"]["disc"],
                                          cfgs_train["loss"]["disc_weighting"],
                                          disc_real=disc_real,
                                          disc_fake=disc_fake)
                disc_loss_sum += disc_loss.item()

                optimizers[1].zero_grad()
                # disc_loss = torch.autograd.Variable(disc_loss, requires_grad=True)
                disc_loss.backward(retain_graph=True)
                optimizers[1].step()

                # train generator
                disc_fake = models[1](data, target_fake)
                gen_loss = loss_function(cfgs_train["loss"]["gen"],
                                         cfgs_train["loss"]["gen_weighting"],
                                         disc_fake=disc_fake,
                                         target_fake=target_fake,
                                         target=target)
                gen_loss_sum += gen_loss.item()

                optimizers[0].zero_grad()
                # gen_loss = torch.autograd.Variable(gen_loss, requires_grad=True)
                gen_loss.backward()
                optimizers[0].step()

                tepoch.set_postfix(loss_gen=f"{round(gen_loss.item() / data.shape[0], 3)}")

    avg_disc_loss = disc_loss_sum / ((len(dataloader)-1)*cfgs_train["batch_size"] + data.shape[0])
    avg_gen_loss = gen_loss_sum / ((len(dataloader)-1)*cfgs_train["batch_size"] + data.shape[0])
    # print(f"Training avg loss: {avg_loss:.2f}.")
    return avg_gen_loss, avg_disc_loss


def validation(models, dataloader, device: str, all_cfgs: dict, save_data=False):
    total_loss = 0
    [model.eval() for model in models]
    enable_dropout(models[0])
    cfgs_train = all_cfgs["train"]
    metrics_names = all_cfgs["metrics"]
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Validation epoch [{cfgs_train['epoch']}/{cfgs_train['epochs']}]")
            metrics = {metric: 0 for metric in metrics_names}
            metrics_ratio = {metric: 0 for metric in metrics_names}
            for idx, (data, target) in enumerate(tepoch):
                data, target = data.to(device).float(), target.to(device).float()

                target_fake = models[0](data)
                if save_data:
                    save_input_output_pairs(data, target_fake, all_cfgs, idx)
                mask = torch.where(target != 0, 1, 0)
                target_fake = torch.mul(target_fake.detach(), mask).float()
                disc_fake = models[1](data, target_fake)
                gen_loss = loss_function(cfgs_train["loss"]["gen"],
                                         cfgs_train["loss"]["gen_weighting"],
                                         disc_fake=disc_fake,
                                         target_fake=target_fake,
                                         target=target)
                total_loss += gen_loss.item()

                # get metrics and ratio of metrics for generated outputs
                metric_values = get_metrics(metrics_names, target, target_fake)
                metric_values_baseline = get_metrics(metrics_names, target, data)
                if metric_values_baseline["rmse"] == 0:
                    print("RMSE of baseline is 0!")
                if metric_values_baseline["vector_correlation"] == 2:
                    print("Vector correlation of baseline is 2!")
                for metric_name in metrics_names:
                    metrics[metric_name] += metric_values[metric_name]
                    metrics_ratio[metric_name] += metric_values[metric_name] / (metric_values_baseline[metric_name]+1e-8)

                tepoch.set_postfix(loss=str(round(gen_loss.item() / data.shape[0], 3)))

    metrics = {metric_name: metric_value/len(dataloader) for metric_name, metric_value in metrics.items()}
    metrics_ratio = {metric_name + "_ratio": metric_value/len(dataloader) for metric_name, metric_value in metrics_ratio.items()}
    avg_loss = total_loss / ((len(dataloader)-1)*cfgs_train["batch_size"] + data.shape[0])
    # print(f"Validation avg loss: {avg_loss:.2f}")
    return avg_loss, metrics, metrics_ratio


def clean_up_training(models, optimizers, name_extensions, dataloader, all_cfgs: dict, device: str):
    """Saves final model. Saves plot for fixed batch."""

    for i in range(len(models)):
        save_checkpoint(models[i],
                        optimizers[i],
                        f"{os.path.join(all_cfgs['save_base_path'], all_cfgs['model_save_name'].split('.')[0])}_{name_extensions[i]}.pth")
    _ = predict_fixed_batch(models[0], dataloader, device)
    wandb.finish()


def main(sweep: Optional[bool] = False):
    all_cfgs = initialize(sweep=sweep)
    print("####### Start Training #######")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> Running on: {device}.")
    all_cfgs["device"] = device

    # seed for reproducibility
    torch.manual_seed(0)

    # simplify config access
    model_types = all_cfgs["model"]
    cfgs_gen = all_cfgs[model_types[0]]
    cfgs_disc = all_cfgs[model_types[1]]
    cfgs_dataset = all_cfgs["dataset"]
    cfgs_train = all_cfgs["train"]
    cfgs_gen_optimizer = all_cfgs["gen_optimizer"]
    cfgs_disc_optimizer = all_cfgs["disc_optimizer"]
    cfgs_lr_scheduler = all_cfgs["train"]["lr_scheduler_configs"]

    # load training data
    train_loader, val_loader, fixed_batch_loader = get_data(all_cfgs["dataset_type"], cfgs_dataset, cfgs_train)

    # define model and optimizer and load from checkpoint if specified
    print(f"-> Model: {model_types}.")
    print(f"-> Gen Losses: {cfgs_train['loss']['gen']} with weightings {cfgs_train['loss']['gen_weighting']}.")
    print(f"-> Gen Losses: {cfgs_train['loss']['disc']} with weightings {cfgs_train['loss']['disc_weighting']}.")

    generator = get_model(model_types[0], cfgs_gen, device)
    discriminator = get_model(model_types[1], cfgs_disc, device)
    gen_optimizer = get_optimizer(generator, cfgs_gen_optimizer["name"],
                                  cfgs_gen_optimizer["parameters"],
                                  lr=cfgs_train["learning_rate"])
    disc_optimizer = get_optimizer(discriminator, cfgs_disc_optimizer["name"],
                                   cfgs_disc_optimizer["parameters"],
                                   lr=cfgs_train["learning_rate"])
    if cfgs_lr_scheduler["value"]:
        gen_lr_scheduler = get_scheduler(gen_optimizer, cfgs_lr_scheduler)
        disc_lr_scheduler = get_scheduler(disc_optimizer, cfgs_lr_scheduler)

    # load models/instantiate models
    if cfgs_gen["load_from_chkpt"]:
        gen_checkpoint_path = os.path.join(all_cfgs["save_base_path"], cfgs_gen["chkpt"])
        load_checkpoint(gen_checkpoint_path, generator, gen_optimizer, cfgs_train["learning_rate"], device)
        if cfgs_gen["init_decoder"]:
            with torch.no_grad():
                for name, layer in generator.named_modules():
                    # print(name, type(layer))
                    if ("up" in name and "conv.0" in name) or "final_up.0" in name:
                        print(name, type(layer))
                        nn.init.xavier_normal_(layer.weight.data, gain=0.02)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias.data, 0.0)
    else:
        init_weights(generator, init_type=cfgs_gen["init_type"], init_gain=cfgs_gen["init_gain"])
    if cfgs_disc["load_from_chkpt"]:
        disc_checkpoint_path = os.path.join(all_cfgs["save_base_path"], all_cfgs["chkpt"])
        load_checkpoint(disc_checkpoint_path, discriminator, disc_optimizer, cfgs_train["learning_rate"], device)
    else:
        init_weights(discriminator, init_type=cfgs_disc["init_type"], init_gain=cfgs_disc["init_gain"])

    # freeze encoder of generator
    if cfgs_gen["freeze_encoder"]:
        for name, param in generator.named_parameters():
            if name.find("down") != -1 or name.find("bottleneck") != -1:
                param.requires_grad = False
            else:
                param.requires_grad = True
        print("-> frozen generator encoder")

    # torch.onnx.export(model, torch.randn(1, 2, 256, 256), "/home/jonas/Downloads/my_model.onnx")

    train_losses_gen, train_losses_disc, val_losses, lrs = list(), list(), list(), list()
    try:
        for epoch in range(1, cfgs_train["epochs"] + 1):
            print()
            to_log = dict()
            to_log |= {"gen_lr": gen_optimizer.param_groups[0]["lr"],
                       "disc_lr": disc_optimizer.param_groups[0]["lr"]}
            cfgs_train["epoch"] = epoch

            train_loss_gen, train_loss_disc = train((generator, discriminator),
                                                    (gen_optimizer, disc_optimizer),
                                                    train_loader,
                                                    device,
                                                    cfgs_train)
            train_losses_gen.append(train_loss_gen)
            train_losses_disc.append(train_loss_disc)
            to_log |= {"train_loss_gen": train_loss_gen, "train_loss_disc": train_loss_disc}

            if len(val_loader) != 0:
                val_loss, metrics, metrics_ratio = validation((generator, discriminator),
                                                              val_loader,
                                                              device,
                                                              all_cfgs)
                val_losses.append(val_loss)
                to_log |= {"val_loss": val_loss}
                to_log |= metrics
                to_log |= metrics_ratio
            if cfgs_lr_scheduler["value"]:
                gen_lr_scheduler.step(val_loss)
                disc_lr_scheduler.step(val_loss)

            if all_cfgs["save_model"] and epoch % 5 == 0:
                gen_checkpoint_path = os.path.join(all_cfgs["save_base_path"], f"{all_cfgs['model_save_name']}_gen.pth")
                save_checkpoint(generator, gen_optimizer, gen_checkpoint_path)
                disc_checkpoint_path = os.path.join(all_cfgs["save_base_path"], f"{all_cfgs['model_save_name']}_disc.pth")
                save_checkpoint(discriminator, disc_optimizer, disc_checkpoint_path)

            visualisations = predict_fixed_batch(generator, fixed_batch_loader, device)
            to_log |= visualisations

            wandb.log(to_log)

    finally:
        clean_up_training((generator, discriminator),
                          (gen_optimizer, disc_optimizer),
                          ("gen", "disc"),
                          fixed_batch_loader,
                          all_cfgs,
                          device)


def test():
    all_cfgs = initialize(sweep=False, test=True)
    print("####### Start Testing #######")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # simplify config access
    model_types = all_cfgs["model"]
    cfgs_gen = all_cfgs[model_types[0]]
    cfgs_disc = all_cfgs[model_types[1]]
    cfgs_dataset = all_cfgs["test_dataset"]
    cfgs_train = all_cfgs["train"]

    # load training data
    test_loader = get_test_data(all_cfgs["dataset_type"], cfgs_dataset, cfgs_train)

    # load generator
    gen = get_model(all_cfgs["model"][0], cfgs_gen, device)
    checkpoint_path = os.path.join(all_cfgs["save_base_path"],
                                   f"{all_cfgs['test_load_chkpt'].split('.')[0]}_gen.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(checkpoint["state_dict"])

    # load discriminator
    disc = get_model(all_cfgs["model"][1], cfgs_disc, device)
    checkpoint_path = os.path.join(all_cfgs["save_base_path"],
                                   f"{all_cfgs['test_load_chkpt'].split('.')[0]}_disc.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    disc.load_state_dict(checkpoint["state_dict"])

    cfgs_train["epoch"] = 1
    _, metrics, _ = validation((gen, disc), test_loader, device, all_cfgs, save_data=True)
    print(metrics)


if __name__ == "__main__":
    # main()
    test()
