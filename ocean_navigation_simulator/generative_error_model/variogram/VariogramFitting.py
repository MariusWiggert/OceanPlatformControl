import numpy as np
import matplotlib.pyplot as plt
from typing import List
import pandas as pd
from lmfit.models import Model
import lmfit


class VariogramFitting:

    def __init__(self, data: pd.DataFrame, error_var: str="u_semivariance"):
        # ensure no zeros in data (by product of Variogram creation)
        self.data = data[data[error_var] != 0]
        self.error_var = error_var
        self.lags = np.hstack((self.data["lon_lag"].to_numpy().reshape(-1, 1),
                               self.data["lat_lag"].to_numpy().reshape(-1, 1),
                               self.data["time_lag"].to_numpy().reshape(-1, 1)))

    def fit_model(self, num_stacked_functions):
        self.model = Model(multi_dim_gaussian, prefix=f"g0_")
        for i in range(1, num_stacked_functions):
            self.model += Model(multi_dim_gaussian, prefix=f"g{i}_")
        print(f"Number of models: {len(self.model.components)}")

        params = self.initialise_parameters()
        result = self.model.fit(self.data[self.error_var], params, lag_vec=self.lags)

        self.popt = np.array(list(result.params.valuesdict().values())).reshape(-1, 4)
        print(f"Parameters:\n {self.popt}")

    def initialise_parameters(self, contraint_weighting=False):
        params = lmfit.Parameters()
        weights_list = []
        num_components = len(self.model.components)
        for i, component in enumerate(self.model.components):
            if contraint_weighting:
                if i == num_components-1:
                    params.add(f"{component.prefix}s", value=np.random.rand(), expr=f"1.0 - {' - '.join(weights_list)}", max=1, min=0)
                else:
                    params.add(f"{component.prefix}s", value=np.random.rand(), max=1, min=0)
                weights_list.append(f"{component.prefix}s")
            else:
                params.add(f"{component.prefix}s", value=np.random.rand(), max=1, min=0)
                params.add(f"{component.prefix}r_lon", value=np.random.rand() * 1000, max=2000, min=10)
                params.add(f"{component.prefix}r_lat", value=np.random.rand() * 1000, max=2000, min=10)
                params.add(f"{component.prefix}r_t", value=np.random.rand() * 1000, max=2000, min=10)
        return params

    def visualize_sliced_variogram(self, var: str="lon_lag"):
        range_param_map = {"lon_lag": 1, "lat_lag": 2, "time_lag": 3}
        # compute lags in var
        var_lags = list(range(sorted(list(set(self.data[var])))[-1]))

        # add contributions from multiple stacked functions
        var_semivariance = []
        for component in (range(self.popt.shape[0])):
            weighting = self.popt[component, 0]
            range_param = self.popt[component, range_param_map[var]]
            if component == 0:
                var_semivariance = np.array(self._get_values_from_fitted_function(var_lags, weighting, range_param))
            else:
                var_semivariance += np.array(self._get_values_from_fitted_function(var_lags, weighting, range_param))

        # slice variogram data to get empirical points
        index_to_remove = range_param_map[var] - 1
        res = [self.data["lon_lag"].min(), self.data["lat_lag"].min(), self.data["time_lag"].min()]
        vars = list(range_param_map.keys())
        del vars[index_to_remove]
        del res[index_to_remove]
        # filter for points where other variables are in smallest bin
        var_slice = self.data[(self.data[vars[0]] == res[0]) & (self.data[vars[1]] == res[1])]
        # select column of interested variable
        var_lags_variogram = var_slice[var]
        # select correct error {u,v}
        var_semivariance_variogram = var_slice[self.error_var]

        # visualize slices
        _ = plt.figure(figsize=(20, 12))
        plt.plot(var_lags, var_semivariance, label="fitted function")
        plt.scatter(var_lags_variogram, var_semivariance_variogram, label="empirical points")
        plt.legend()
        plt.xlim(left=0)
        plt.ylim([0, 1.5])
        plt.xlabel(f"{var}")
        plt.ylabel("semivariance")

    def _get_values_from_fitted_function(self, var_lags: List[float], weighting: float, range_param: float):
        semivariance = []
        for lag in var_lags:
            semivariance.append(self._multi_dim_gaussian_slice(lag, weighting, range_param))
        return semivariance

    def _multi_dim_gaussian_slice(self, lag, s, r):
        """1D version of multi_dim_gaussian for visualizing slices."""
        gamma = s*(1-np.exp(-3*(1/r**2)*lag**2))
        return gamma


def multi_dim_gaussian(lag_vec, s, r_lon, r_lat, r_t):
    """3D Gaussian similar to whatBradley defined in pre-print."""
    h = lag_vec
    Omega = np.diag([1/(r_lon)**2, 1/(r_lat)**2, 1/(r_t)**2])
    h_Omega = np.dot(h, Omega)
    # row-wise dot product of two matrices
    # https://stackoverflow.com/questions/15616742/vectorized-way-of-calculating-row-wise-dot-product-two-matrices-with-scipy
    exponent = np.sum(h_Omega*h, axis=1)
    gamma = s*(1 - exp_normalize(-3*exponent))
    return gamma


def exp_normalize(x):
    """More numerically stable exponential function."""
    b = x.max()
    y = np.exp(x-b)
    return y / y.sum()
