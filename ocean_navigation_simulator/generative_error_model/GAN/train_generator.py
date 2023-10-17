import os
import sys
from typing import Dict, List, Optional
from warnings import warn

import torch
import torch.nn as nn
import yaml
from torchvision.utils import make_grid
from tqdm import tqdm

import wandb
from ocean_navigation_simulator.generative_error_model.GAN.helper_funcs import (
    enable_dropout,
    get_data,
    get_model,
    get_optimizer,
    get_scheduler,
    get_test_data,
    initialize,
    save_input_output_pairs,
)
from ocean_navigation_simulator.generative_error_model.GAN.ssim import ssim
from ocean_navigation_simulator.generative_error_model.GAN.utils import (
    init_weights,
    l1,
    load_checkpoint,
    mass_conservation,
    mse,
    save_checkpoint,
    sparse_mse,
    total_variation,
)
from ocean_navigation_simulator.generative_error_model.generative_model_metrics import (
    rmse,
    vector_correlation,
)


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
    return {
        "fixed_batch_gt": ground_truth,
        "fixed_batch_predictions": predictions,
        "fixed_batch_inputs": samples,
    }


def get_metrics(
    metric_names, ground_truth: torch.Tensor, predictions: torch.Tensor
) -> Optional[Dict[str, float]]:
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
            rmse_val = rmse(
                ground_truth[idx, :2, :, :].squeeze(), predictions[idx, :2, :, :].squeeze()
            )
            metrics["rmse"] += rmse_val
        if "vector_correlation" in metric_names:
            vec_corr_val = vector_correlation(
                ground_truth[idx, 0, :, :].squeeze(),
                ground_truth[idx, 1, :, :].squeeze(),
                predictions[idx, 0, :, :].squeeze(),
                predictions[idx, 1, :, :].squeeze(),
            )
            metrics["vector_correlation"] += vec_corr_val
        if "ssim" in metric_names:
            ssim_val = ssim(
                torch.Tensor(ground_truth[idx].squeeze()), torch.Tensor(predictions[idx].squeeze())
            )
            metrics["ssim"] += ssim_val
    metrics = {
        metric_name: metric_value / ground_truth.shape[0]
        for metric_name, metric_value in metrics.items()
    }
    return metrics


def loss_function(predictions, target, losses: List[str], loss_weightings: List[float]):
    """Handles which loss is to be used."""
    loss = 0
    loss_map = {
        "mse": mse,
        "l1": l1,
        "sparse_mse": sparse_mse,
        "total_variation": total_variation,
        "mass_conservation": mass_conservation,
    }

    if len(losses) != len(loss_weightings):
        raise ValueError("'losses' and 'loss_weightings' are not the same length!")

    for loss_type, weight in zip(losses, loss_weightings):
        if loss_type in ["mse", "l1", "sparse_mse"]:
            loss += weight * loss_map[loss_type](predictions, target)
        else:
            loss += weight * loss_map[loss_type](predictions)

    if loss == 0:
        raise warn("Loss is zero!")
    return loss


def train(model: nn.Module, dataloader, device, optimizer, cfgs_train):
    total_loss = 0
    model.train()
    with torch.enable_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Training epoch [{cfgs_train['epoch']}/{cfgs_train['epochs']}]")
            for data, target in tepoch:
                data, target = data.to(device), target.to(device)

                output = model(data)

                # compute loss
                train_loss = loss_function(
                    output, target, cfgs_train["loss"]["gen"], cfgs_train["loss"]["gen_weighting"]
                )
                total_loss += train_loss.item()

                # perform optim step
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=str(round(train_loss.item() / data.shape[0], 3)))

    avg_loss = total_loss / ((len(dataloader) - 1) * cfgs_train["batch_size"] + data.shape[0])
    # print(f"Training avg loss: {avg_loss:.2f}.")
    return avg_loss


def validation(model, dataloader, device: str, all_cfgs: dict, save_data=False):
    total_loss = 0
    model.eval()
    cfgs_train = all_cfgs["train"]
    metrics_names = all_cfgs["metrics"]
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(
                f"Validation epoch [{cfgs_train['epoch']}/{cfgs_train['epochs']}]"
            )
            metrics = {metric: 0 for metric in metrics_names}
            metrics_ratio = {metric: 0 for metric in metrics_names}
            for idx, (data, target) in enumerate(tepoch):
                data, target = data.to(device), target.to(device)

                output = model(data)
                if save_data:
                    save_input_output_pairs(data, output, all_cfgs, idx)
                val_loss = loss_function(
                    output, target, cfgs_train["loss"]["gen"], cfgs_train["loss"]["gen_weighting"]
                )
                total_loss += val_loss.item()

                # get metrics and ratio of metrics for generated outputs
                metric_values = get_metrics(metrics_names, target, output)
                metric_values_baseline = get_metrics(metrics_names, target, data)
                # if metric_values_baseline["rmse"] == 0:
                #     print("RMSE of baseline is 0!")
                # if metric_values_baseline["vector_correlation"] == 2:
                #     print("Vector correlation of baseline is 2!")
                for metric_name in metrics_names:
                    metrics[metric_name] += metric_values[metric_name]
                    metrics_ratio[metric_name] += metric_values[metric_name] / (
                        metric_values_baseline[metric_name] + 1e-8
                    )

                tepoch.set_postfix(loss=str(round(val_loss.item() / data.shape[0], 3)))

    metrics = {
        metric_name: metric_value / len(dataloader) for metric_name, metric_value in metrics.items()
    }
    metrics_ratio = {
        metric_name + "_ratio": metric_value / len(dataloader)
        for metric_name, metric_value in metrics_ratio.items()
    }
    avg_loss = total_loss / ((len(dataloader) - 1) * cfgs_train["batch_size"] + data.shape[0])
    # print(f"Validation avg loss: {avg_loss:.2f}")
    return avg_loss, metrics, metrics_ratio


def clean_up_training(model, optimizer, dataloader, all_cfgs: dict, device: str):
    """Saves final model. Saves plot for fixed batch."""

    save_checkpoint(
        model, optimizer, f"{os.path.join(all_cfgs['save_base_path'], all_cfgs['model_save_name'])}"
    )
    _ = predict_fixed_batch(model, dataloader, device)
    wandb.finish()


def train_main(sweep: Optional[bool] = False):
    all_cfgs = initialize(sweep=sweep)
    print("####### Start Training #######")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> Running on: {device}.")
    all_cfgs["device"] = device

    # seed for reproducibility
    torch.manual_seed(0)

    # simplify config access
    model_type = all_cfgs["model"]
    cfgs_model = all_cfgs[model_type]
    cfgs_dataset = all_cfgs["dataset"]
    cfgs_train = all_cfgs["train"]
    cfgs_optimizer = all_cfgs["optimizer"]
    cfgs_lr_scheduler = all_cfgs["train"]["lr_scheduler_configs"]

    # load training data
    train_loader, val_loader, fixed_batch_loader = get_data(
        all_cfgs["dataset_type"], cfgs_dataset, cfgs_train
    )

    # define model and optimizer and load from checkpoint if specified
    print(f"-> Model: {model_type}.")
    print(
        f"-> Losses: {cfgs_train['loss']['gen']} with weightings {cfgs_train['loss']['gen_weighting']}."
    )
    model = get_model(model_type, cfgs_model, device)
    optimizer = get_optimizer(
        model, cfgs_optimizer["name"], cfgs_optimizer["parameters"], lr=cfgs_train["learning_rate"]
    )
    if cfgs_lr_scheduler["value"]:
        lr_scheduler = get_scheduler(optimizer, cfgs_lr_scheduler)
    if all_cfgs["load_from_chkpt"]["value"]:
        checkpoint_path = os.path.join(
            all_cfgs["save_base_path"], all_cfgs["load_from_chkpt"]["file_name"]
        )
        load_checkpoint(checkpoint_path, model, optimizer, cfgs_train["learning_rate"], device)
    else:
        init_weights(model, init_type=cfgs_model["init_type"], init_gain=cfgs_model["init_gain"])
    # torch.onnx.export(model, torch.randn(1, 2, 256, 256), "/home/jonas/Downloads/my_model.onnx")

    train_losses, val_losses = list(), list()
    try:
        for epoch in range(1, cfgs_train["epochs"] + 1):
            print()
            to_log = dict()
            to_log |= {"lr": optimizer.param_groups[0]["lr"]}
            cfgs_train["epoch"] = epoch

            train_loss = train(model, train_loader, device, optimizer, cfgs_train)
            train_losses.append(train_loss)
            to_log |= {"train_loss": train_loss}

            if len(val_loader) != 0:
                val_loss, metrics, metrics_ratio = validation(
                    model, val_loader, device, all_cfgs, save_data=False
                )
                val_losses.append(val_loss)
                to_log |= {"val_loss": val_loss}
                to_log |= metrics
                to_log |= metrics_ratio
            if cfgs_lr_scheduler["value"]:
                lr_scheduler.step(val_loss)

            if epoch % 1 == 0 or epoch == 1:
                visualisations = predict_fixed_batch(model, fixed_batch_loader, device)
                to_log |= visualisations

            wandb.log(to_log)

    finally:
        clean_up_training(model, optimizer, fixed_batch_loader, all_cfgs, device)


def test_main(data: str = "test"):
    all_cfgs = initialize(sweep=False, test=True)
    print("####### Start Testing #######")
    print(f"In mode: {data}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # simplify config access
    cfgs_model = all_cfgs[all_cfgs["model"]]
    cfgs_train = all_cfgs["train"]

    if data == "test":
        # load test_data
        cfgs_dataset = all_cfgs["test_dataset"]
    elif data == "val":
        cfgs_dataset = all_cfgs["val_dataset"]
        cfgs_train["batch_size"] = 192
    else:
        raise ValueError(f"data = {data} is not a valid input! Try: ['test', 'val'].")
    dataloader = get_test_data(all_cfgs["dataset_type"], cfgs_dataset, cfgs_train)

    # load model
    model = get_model(all_cfgs["model"], cfgs_model, device)
    checkpoint_path = os.path.join(
        all_cfgs["save_base_path"], all_cfgs["load_from_chkpt"]["file_name"]
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    model.eval()
    if all_cfgs["generator"]["dropout"] is False:
        enable_dropout(model, all_cfgs["validation"]["layers"])
    save_dirs = []
    repeated_data = None
    # iterate twice: once for all test/val FCs, once for repeated FC
    for loader_idx in range(2):
        with torch.no_grad():
            for idx, (data, _) in enumerate(tqdm(dataloader)):
                data = data.to(device).float()
                # repeated sample
                if loader_idx == 1:
                    if idx == 0:
                        repeated_data = data
                    if idx == 10:
                        break
                    target_fake = model(repeated_data)
                    save_dir = save_input_output_pairs(
                        repeated_data,
                        target_fake,
                        all_cfgs,
                        all_cfgs["save_repeated_samples_path"],
                        idx,
                    )
                # normal samples
                else:
                    target_fake = model(data)
                    save_dir = save_input_output_pairs(
                        data, target_fake, all_cfgs, all_cfgs["save_samples_path"], idx
                    )
        save_dirs.append(save_dir)
    return save_dirs


def main():
    config_file = sys.argv[1]
    all_cfgs = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    if all_cfgs["run_train"]:
        train_main()
    else:
        print(test_main(data="test"))


if __name__ == "__main__":
    main()
