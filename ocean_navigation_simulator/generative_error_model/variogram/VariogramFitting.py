import ast
import os
from typing import List, Tuple

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit.models import Model


class VariogramFitting:
    """This class takes a tuned empirical variogram as input data and fits multiple stacked
    Gaussian component models to it, together these make up the overall model.

    The purpose of this class is to output a weight and a range vector for each component.
    These output parameters are subsequently used to tune the simplex noise.
    """

    def __init__(self, data: pd.DataFrame, lag_vars: Tuple, error_var: str = "u_semivariance"):
        # ensure no zeros in data (by product of Variogram creation)
        self.data = data[data[error_var] != 0]
        self.error_var = error_var
        self.lag_vars = lag_vars
        self.range_vars = None
        self.model = None
        self.num_components = None
        self.popt = None

        # Get lags in right format
        lags = []
        for lag_var in lag_vars:
            lags.append(self.data[lag_var].to_numpy().reshape(-1, 1))
        self.lags = np.hstack(tuple(lags))

        # Converting string to list
        detrend_u = ast.literal_eval(data["detrend_u"][0])
        detrend_v = ast.literal_eval(data["detrend_v"][0])
        self.detrend_metrics = [detrend_u, detrend_v]

    def fit_model(
        self,
        num_of_components: int,
        method: str = "leastsq",
        constrain_weighting: bool = False,
        verbose: bool = False,
    ):
        """Uses the lags and their associated values and fits multiple stacked gaussian models to it.
        The output is of size [n,d]:
            n - number of harmonics
            d - lag space dimension
        One harmonic consists of [weighting, [range vector]]:
            range vector [d-1], either in R^3 or R^2
        """
        if len(self.lag_vars) == 3:
            model_type = gaussian_3d
            self.range_vars = ["r_lon", "r_lat", "r_t"]
        else:
            model_type = gaussian_2d
            self.range_vars = ["r_space", "r_t"]
        self.num_components = num_of_components
        self.model = Model(model_type, prefix="g0_")
        for i in range(1, num_of_components):
            self.model += Model(model_type, prefix=f"g{i}_")
        print(f"Number of models: {len(self.model.components)}")
        print(f"Type of model: {model_type.__name__}")

        params = self.initialise_parameters(constrain_weighting=constrain_weighting)
        # weights_1d = np.array(list(np.logspace(0.1, 100, self.lags.shape[0]))[::-1])
        # weights = weights_1d
        weights = list(np.ones(self.lags.shape[0]))
        # fit model: data points, parameters, lags for the data points (dependent variables)
        result = self.model.fit(
            self.data[self.error_var],
            params=params,
            lag_vec=self.lags,
            method=method,
            weights=weights,
        )

        self.popt = np.array(list(result.params.valuesdict().values())).reshape(
            -1, len(self.lag_vars) + 1
        )
        print(f"Parameters:\n {self.popt}")
        if verbose:
            print(result.fit_report())

    def initialise_parameters(self, constrain_weighting=False):
        """Initializes every parameter for every component of the model."""
        params = lmfit.Parameters()
        weights_list = []
        for i, component in enumerate(self.model.components):
            if constrain_weighting:
                # constrain sum of weights to 1.
                if i == self.num_components - 1:
                    params.add(
                        f"{component.prefix}s",
                        value=1 / self.num_components,
                        expr=f"1.0 - {' - '.join(weights_list)}",
                        max=1,
                        min=0,
                    )
                else:
                    params.add(f"{component.prefix}s", value=1 / self.num_components, max=1, min=0)
                for range_var in self.range_vars:
                    params.add(
                        f"{component.prefix}{range_var}",
                        value=np.random.rand() * 300,
                        max=1000,
                        min=20,
                    )
                weights_list.append(f"{component.prefix}s")
            else:
                # no contraints
                params.add(f"{component.prefix}s", value=1 / self.num_components, max=1, min=0)
                for range_var in self.range_vars:
                    params.add(
                        f"{component.prefix}{range_var}",
                        value=np.random.rand() * 300,
                        max=1000,
                        min=20,
                    )
        return params

    def plot_all_dims(self, save_path: str = None, plot_empirical: bool = True):
        """Plot the sliced fitted function over all lag variables."""
        figure, axs = plt.subplots(1, len(self.lag_vars), figsize=(15, 6))
        units_map = {"space_lag": "km", "t_lag": "hrs", "lon_lag": "km", "lat_lag": "km"}
        for idx, var in enumerate(self.lag_vars):
            self._plot_sliced_variogram(
                var, units_map[var], axs[idx], plot_empirical=plot_empirical
            )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, pad_inches=0, facecolor="white")
        return figure, axs

    def load_params(self, input_path: str) -> None:
        """Load previously exported parameters from file."""
        params = np.load(input_path, allow_pickle=True)
        params_map = {"u_semivariance": "U_COMP", "v_semivariance": "V_COMP"}
        params = np.array(params.item().get(params_map[self.error_var]))
        model_type = gaussian_3d
        self.range_vars = ["r_lon", "r_lat", "r_t"]
        if len(self.lag_vars) == 2:
            params = params[:, [0, 1, 3]]  # remove doubled space axis
            model_type = gaussian_2d
            self.range_vars = ["r_space", "r_t"]

        # create model and add require number of components.
        self.model = Model(model_type, prefix="g0_")
        for i in range(1, params.shape[0]):
            self.model += Model(model_type, prefix=f"g{i}_")
        print(f"Number of models: {len(self.model.components)}")
        print(f"Type of model: {model_type.__name__}")

        # initialize parameters
        model_params = lmfit.Parameters()
        for i, component in enumerate(self.model.components):
            model_params.add(f"{component.prefix}s", value=params[i, 0])
            for j, range_var in enumerate(self.range_vars):
                model_params.add(f"{component.prefix}{range_var}", value=params[i, j + 1])
        self.popt = params

    def save_params(self, output_path: str) -> None:
        """For now this saved the same the same parameters for 'u and 'v'."""
        params = self.popt
        if len(self.lag_vars) == 2:
            # need to convert 2d params to 3d
            params = np.hstack(
                (
                    params[:, 0].reshape(-1, 1),
                    params[:, 1].reshape(-1, 1),
                    params[:, 1].reshape(-1, 1),
                    params[:, 2].reshape(-1, 1),
                )
            )

        param_map = {"u": "U_COMP", "v": "V_COMP"}
        component = param_map[self.error_var[0]]
        other_component = list(param_map.values())
        other_component.remove(component)
        other_component = other_component[0]
        params_dict = {"U_COMP": [], "V_COMP": []}

        if os.path.exists(output_path):
            # if file already exists -> update file with current popt params
            loaded_params = np.load(output_path, allow_pickle=True)
            detrend_metrics = loaded_params.item().get("detrend_metrics")
            other_params = loaded_params.item().get(other_component)
            for i in range(params.shape[0]):
                params_dict[component].append(params[i])
            params_dict[other_component] = other_params
            params_dict["detrend_metrics"] = detrend_metrics
        else:
            # if file does not exist write new file
            for i in range(params.shape[0]):
                params_dict[component].append(params[i])
            params_dict["detrend_metrics"] = self.detrend_metrics

        np.save(output_path, params_dict)
        print(f"Saved {params_dict}")

    def _plot_sliced_variogram(
        self, var: str, units: str, ax: plt.axis, plot_empirical: bool = True
    ) -> plt.axis:
        """Plots the fitted function vs the empirical data points."""
        if var not in self.lag_vars:
            raise ValueError(f"Your specified var='{var}' is not in {self.lag_vars}!")

        # get data for fitted function line plot
        lags, function_semivariance = self._get_sliced_fitted_function_data(var)
        ax.plot(lags, function_semivariance, label="variogram model", color="orange")

        if plot_empirical:
            # get data for empirical points scatter plot
            empirical_lags, empirical_semivariance = self._get_sliced_empirical_data(var)
            ax.scatter(empirical_lags, empirical_semivariance, marker="x", label="empirical points")

        ax.legend()
        ax.set_xlim(left=0)
        ax.set_ylim([0, 1.5])
        ax.set_xlabel(f"{var} [{units}]")
        ax.set_ylabel("semivariance")
        return ax

    def _get_sliced_fitted_function_data(self, var: str):
        """Uses the computed parameters to slice the function in the variable specified."""
        # TODO: Maybe use self.model.eval() here which is part of lmfit
        lag_idx = self.lag_vars.index(var)
        # need to create set of lags because there will be multiples
        lags = sorted(list(set(self.lags[:, lag_idx])))
        semivariance = []
        for component in range(self.popt.shape[0]):
            weighting = self.popt[component, 0]
            range_param = self.popt[component, self.lag_vars.index(var) + 1]
            if component == 0:
                semivariance = np.array(
                    self._get_values_from_fitted_function(lags, weighting, range_param)
                )
            else:
                semivariance += np.array(
                    self._get_values_from_fitted_function(lags, weighting, range_param)
                )
        return lags, semivariance

    def _get_sliced_empirical_data(self, var: str):
        """Slices the empirical data to get data for the variable specified."""
        min_res = []
        for lag_var in self.lag_vars:
            min_res.append(self.data[lag_var].min())
        lag_vars = list(self.lag_vars)
        # remove the lag we want to plot
        index_to_remove = self.lag_vars.index(var)
        del lag_vars[index_to_remove]
        del min_res[index_to_remove]
        # filter for points where other variables are in smallest bin
        if len(lag_vars) == 2:
            var_slice = self.data[
                (self.data[lag_vars[0]] == min_res[0]) & (self.data[lag_vars[1]] == min_res[1])
            ]
        else:
            var_slice = self.data[(self.data[lag_vars[0]] == min_res[0])]
        # select column of interested variable
        var_lags_variogram = var_slice[var]
        # select correct error {u,v}
        var_semivariance_variogram = var_slice[self.error_var]
        return var_lags_variogram, var_semivariance_variogram

    def _get_values_from_fitted_function(
        self, lags: List[float], weighting: float, range_param: float
    ):
        """Takes a list of lags (for one specific lag dimension) and the appropriate model parameters
        to compute the semivariance.
        """
        semivariance = []
        for lag in lags:
            semivariance.append(self._gaussian_slice(lag, weighting, range_param))
        return semivariance

    def _gaussian_slice(self, lag, s, r):
        """1D version of gaussian function for visualizing slices."""
        gamma = s * (
            1
            - np.exp(-3 * (1 / r**2) * lag**2)
            + 0.06 * np.exp(-7.07 * (1 / r**2) * lag**2)
        )
        return gamma


def gaussian_3d(lag_vec, s, r_lon, r_lat, r_t):
    """3D Gaussian function similar to what Bradley defined in pre-print.
    Args:
        lag_vec: lag vector [n, 3]
        s: sill (weighting)
        r_lon: range in longitudinal direction
        r_lat: range in latitudinal direction
        r_t: range in time direction

    Note: because the lag_vec contains multiple (n many) lag triples the function was re-formulated to vectorize
    the operation 'h.T Omega h' needed in the exponent.
    """
    h = lag_vec
    Omega = np.diag([1 / r_lon**2, 1 / r_lat**2, 1 / r_t**2])
    h_Omega = np.dot(h, Omega)
    # row-wise dot product of two matrices
    # https://stackoverflow.com/questions/15616742/vectorized-way-of-calculating-row-wise-dot-product-two-matrices-with-scipy
    exponent = np.sum(h_Omega * h, axis=1)
    gamma = s * (1 - exp_normalize(-3 * exponent) + 0.06 * exp_normalize(-7.07 * exponent))
    return gamma


def gaussian_2d(lag_vec, s, r_space, r_t):
    """2D Gaussian function similar to what Bradley defined in pre-print.
    Args:
        lag_vec: lag vector
        s: sill (weighting)
        r_space: range in space direction (Euclidean of lon and lat)
        r_t: range in time direction
    """
    h = lag_vec
    Omega = np.diag([1 / r_space**2, 1 / r_t**2])
    h_Omega = np.dot(h, Omega)
    # row-wise dot product of two matrices
    # https://stackoverflow.com/questions/15616742/vectorized-way-of-calculating-row-wise-dot-product-two-matrices-with-scipy
    exponent = np.sum(h_Omega * h, axis=1)
    gamma = s * (1 - exp_normalize(-3 * exponent) + 0.06 * exp_normalize(-7.07 * exponent))
    return gamma


def exp_normalize(x):
    """More numerically stable exponential function."""
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()
