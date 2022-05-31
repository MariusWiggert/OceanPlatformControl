from __future__ import annotations

from typing import Dict, Callable, Any, Optional, Tuple

import numpy as np
from numpy import ndarray

from ocean_navigation_simulator.utils.calc_fmrc_error import calc_vector_corr_over_time


def get_metrics() -> Dict[str, Callable[[ndarray, ndarray, ndarray], Dict[float, Any]]]:
    """ Get a dictionary where each key-value pair corresponds to the metric name and the corresponding function that
    computes the metrics with the same signature for each of the function

    Returns:
        That dictionary
    """
    return {"r2": r2, "vector_correlation": vector_correlation, "rmse": rmse}


def r2(ground_truth: ndarray, improved_predictions: ndarray, initial_predictions: ndarray, per_hour: bool = False) -> \
        Dict[str, float]:
    """Compute the r2 coefficient where the numerator is the sum of the squared difference between the ground truth and
       the improved forecast. The denominator is the sum of the squared difference between the ground truth and the
       initial forecast.

    Args:
        ground_truth: ndarray containing the ground truth
        improved_predictions: ndarray containing the improved forecast
        initial_predictions: ndarray containing the initial forecast

    Returns:
        The r2 coefficient as a dictionary
    """
    axis = (1, 2) if per_hour else None
    return {("r2_per_h" if per_hour else "r2"): 1 - ((ground_truth - improved_predictions) ** 2).sum(axis=axis) / (
            (initial_predictions - ground_truth) ** 2).sum(axis=axis)}


def vector_correlation(ground_truth: ndarray, improved_predictions: ndarray, initial_predictions: ndarray,
                       sigma_ratio=0.00001, per_hour: bool = False) -> Dict[str, float]:
    """Compute the vector correlation between 1) ground_truth and improved_predictions, 2) ground_truth and
    initial_predictions and also the ratio between these two values

    Args:
        ground_truth: xarray of the ground truth
        improved_predictions: xarray of the improved forecast
        initial_predictions: xarray of the initial forecast
        sigma_ratio: small number used in the division to avoid the division by 0

    Returns:
        The 3 values as a dictionary
    """
    if per_hour:
        print("vector correlation per hour is not implemented")
        return {}
    correlations = dict()
    correlations["vector_correlation_improved"] = calc_vector_corr_over_time(improved_predictions, ground_truth,
                                                                             sigma_diag=0, remove_nans=True).mean()
    correlations["vector_correlation_initial"] = calc_vector_corr_over_time(initial_predictions, ground_truth,
                                                                            sigma_diag=0, remove_nans=True).mean()
    correlations["vector_correlation_ratio"] = correlations["vector_correlation_improved"] / (correlations[
                                                                                                  "vector_correlation_initial"] + sigma_ratio)
    return correlations


def rmse(ground_truth: ndarray, improved_predictions: ndarray, initial_predictions: ndarray, sigma_ratio=0.00001,
         per_hour: bool = False) -> \
        Dict[str, float]:
    """Compute the rmse between 1) ground_truth and improved_predictions, 2) ground_truth and initial_predictions and
    also the ratio between these two values

    Args:
        ground_truth: xarray of the ground truth
        improved_predictions: xarray of the improved forecast
        initial_predictions: xarray of the initial forecast
        sigma_ratio: small number used in the division to avoid the division by 0

    Returns:
        The 3 values as a dictionary
    """
    rmses = dict()
    extension_str = "_per_h" if per_hour else ""
    axis = (1, 2) if per_hour else None
    rmses["rmse_improved" + extension_str] = __rmse(ground_truth, improved_predictions, axis=axis)
    rmses["rmse_initial" + extension_str] = __rmse(ground_truth, initial_predictions, axis=axis)
    rmses["rmse_ratio" + extension_str] = rmses["rmse_improved" + extension_str] / (
            rmses["rmse_initial" + extension_str] + sigma_ratio)

    return rmses


def __rmse(v1: ndarray, v2: ndarray, axis: Optional[int | Tuple[int, ...]] = None) -> float:
    """ Internal function to compute the rmse

    Args:
        v1: vector 1
        v2: vector 2

    Returns:
        rmse between the two vectors v1 and v2
    """
    return np.sqrt(np.mean((v1 - v2) ** 2, axis=axis))
