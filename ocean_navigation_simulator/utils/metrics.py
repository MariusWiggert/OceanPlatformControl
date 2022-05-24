from typing import Dict, Callable, Any

import numpy as np
import xarray as xr

from ocean_navigation_simulator.utils.calc_fmrc_error import calc_vector_corr_over_time


def get_metrics() -> Dict[str, Callable[[xr, xr, xr], Dict[float, Any]]]:
    return {"r2": r2, "vector_correlation": vector_correlation, "rmse": rmse}


def r2(ground_truth: xr, improved_predictions: xr, initial_predictions: xr) -> Dict[str, float]:
    return {"r2": 1 - ((ground_truth - improved_predictions) ** 2).sum() / (
                (initial_predictions - ground_truth) ** 2).sum()}


def vector_correlation(ground_truth: xr, improved_predictions: xr, initial_predictions: xr, sigma_ratio=0.00001) -> \
        Dict[str, float]:
    correlations = dict()
    correlations["vector_correlation_improved"] = calc_vector_corr_over_time(improved_predictions, ground_truth,
                                                                             sigma_diag=0, remove_nans=True).mean()
    correlations["vector_correlation_initial"] = calc_vector_corr_over_time(initial_predictions, ground_truth,
                                                                            sigma_diag=0, remove_nans=True).mean()
    correlations["vector_correlation_ratio"] = correlations["vector_correlation_improved"] / (correlations[
                                                                                                  "vector_correlation_initial"] + sigma_ratio)
    return correlations


def rmse(ground_truth: xr, improved_predictions: xr, initial_predictions: xr, sigma_ratio=0.00001) -> Dict[str, float]:
    rmses = dict()
    rmses["rmse_improved"] = _rmse(ground_truth, improved_predictions)
    rmses["rmse_initial"] = _rmse(ground_truth, initial_predictions)
    rmses["rmse_ratio"] = rmses["rmse_improved"] / (rmses["rmse_initial"] + sigma_ratio)

    return rmses


def _rmse(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.sqrt(np.mean((v1 - v2) ** 2))
