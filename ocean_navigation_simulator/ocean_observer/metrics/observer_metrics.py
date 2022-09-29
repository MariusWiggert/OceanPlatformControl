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
    return {"r2": r2, "vector_correlation": vector_correlation, "rmse": rmse, "vme": vme,
            "ratio_per_tile": ratio_per_tile}


def __get_axis_current(current) -> Tuple[list[int], str]:
    if current == "u":
        return [0], "_u"
    if current == "v":
        return [1], "_v"
    return [0, 1], ""


def r2(ground_truth: ndarray, improved_predictions: ndarray, initial_predictions: ndarray, per_hour: bool = False,
       sigma_square_division: float = 1e-6, current=['uv']) -> Dict[str, float]:
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
    axis_current, extension_name = __get_axis_current(current)
    return {("r2_per_h" if per_hour else "r2") + extension_name: 1 - np.nansum(
        (ground_truth[..., axis_current] - improved_predictions[..., axis_current]) ** 2, axis=axis) / (
                                                                         np.nansum(
                                                                             ((initial_predictions[..., axis_current] -
                                                                               ground_truth[
                                                                                   ..., axis_current]) ** 2),
                                                                             axis=axis) + sigma_square_division)}


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
         per_hour: bool = False, current=["uv"]) -> \
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
    axis_current, extension_str_2 = __get_axis_current(current)
    extension_str += extension_str_2

    rmses["rmse_improved" + extension_str] = __rmse(ground_truth[..., axis_current],
                                                    improved_predictions[..., axis_current], axis=axis)
    rmses["rmse_initial" + extension_str] = __rmse(ground_truth[..., axis_current],
                                                   initial_predictions[..., axis_current], axis=axis)
    rmses["rmse_ratio" + extension_str] = rmses["rmse_improved" + extension_str] / (
            rmses["rmse_initial" + extension_str] + sigma_ratio)

    return rmses


def vme(ground_truth: ndarray, improved_predictions: ndarray, initial_predictions: ndarray, sigma_ratio=0.00001,
        per_hour: bool = False, current=["uv"]) -> \
        Dict[str, float]:
    """Compute the vme between 1) ground_truth and improved_predictions, 2) ground_truth and initial_predictions and
    also the ratio between these two values

    Args:
        ground_truth: xarray of the ground truth
        improved_predictions: xarray of the improved forecast
        initial_predictions: xarray of the initial forecast
        sigma_ratio: small number used in the division to avoid the division by 0

    Returns:
        The 3 values as a dictionary
    """
    vmes = dict()
    extension_str = "_per_h" if per_hour else ""
    axis = (1, 2) if per_hour else None
    axis_current, extension_str_2 = __get_axis_current(current)
    extension_str += extension_str_2

    vmes["vme_improved" + extension_str] = __vector_magnitude_error(ground_truth[..., axis_current],
                                                                    improved_predictions[..., axis_current],
                                                                    axis=axis)
    vmes["vme_initial" + extension_str] = __vector_magnitude_error(ground_truth[..., axis_current],
                                                                   initial_predictions[..., axis_current], axis=axis)
    vmes["vme_ratio" + extension_str] = vmes["vme_improved" + extension_str] / (
            vmes["vme_initial" + extension_str] + sigma_ratio)

    return vmes


def __rmse(v1: ndarray, v2: ndarray, axis: Optional[int | Tuple[int, ...]] = None) -> float:
    """ Internal function to compute the rmse

    Args:
        v1: vector 1
        v2: vector 2

    Returns:
        rmse between the two vectors v1 and v2
    """
    return np.sqrt(np.nanmean((v1 - v2) ** 2, axis=axis))


def __vector_magnitude_error(v1: ndarray, v2: ndarray, axis: Optional[int | Tuple[int, ...]] = None) -> float:
    """Internal function to compute the vme

    Args:
        v1: vector 1
        v2: vector 2
        axis:

    Returns:
        vme between the two vectors v1 and v2
    """
    return np.nanmean(np.sqrt(np.sum(((v1 - v2) ** 2), axis=-1, keepdims=True)), axis=axis)
    # return np.nanmean(np.sum(np.abs(v1 - v2), axis=-1, keepdims=True), axis=axis)


def ratio_per_tile(ground_truth: ndarray, improved_predictions: ndarray, initial_predictions: ndarray,
                   per_hour: bool = False,
                   sigma_square_division: float = 1e-6, current=['uv']) -> Dict[str, float]:
    """Internal function to compute the vme

        Args:
            v1: vector 1, dim: time x lon*lat x 2
            v2: vector 2, dim: time x lon*lat x 2
            axis:

        Returns:
            ratio per tile between the two matrices v1 and v2
        """
    if per_hour:
        UserWarning("Per hour not implemented by ratio per tile.")
    axis_current = -1
    ratios = ((ground_truth - improved_predictions) ** 2).sum(axis=axis_current) / \
             (((ground_truth - initial_predictions) ** 2).sum(axis=axis_current) + sigma_square_division)
    return np.nanmean(ratios)


def check_nans(ground_truth: ndarray, improved_predictions: ndarray, current=["uv"]) -> bool:
    '''

    Args:
        ground_truth:
        improved_predictions:
        initial_predictions:
        sigma_ratio:
        per_hour:
        current:

    Returns: True if the all the hours are full of nans

    '''
    axis_current, extension_str_2 = __get_axis_current(current)
    if np.any(np.isnan(ground_truth[..., axis_current] - improved_predictions[..., axis_current])):
        def percent_nan(A):
            A = np.nan_to_num(A, 0)
            return 1.0 - ((A != 0).sum() / float(A.size))

        percents = np.array([percent_nan(ground_truth[i, ..., axis_current]) for i in range(len(ground_truth))])
        size = improved_predictions[..., axis_current].size
        # if current == "uv":
        #     print(
        #         f"contain NaNs ({current}): "
        #         f"ground_truth:{(np.isnan(ground_truth[..., axis_current]).sum() / size):.2f}%",
        #         f"\tforecast:{(np.isnan(improved_predictions[..., axis_current]).sum() / size):.2f}%")
        return (percents == 1).all()
    return False
