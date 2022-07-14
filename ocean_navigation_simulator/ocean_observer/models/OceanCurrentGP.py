"""
source: https://github.com/google/balloon-learning-environment/blob/cdb2e582f2b03c41f037bf76142d31611f5e0316/balloon_learning_environment/env/wind_gp.py
"""

from typing import Tuple, Dict, Any

import numpy as np
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Kernel

from ocean_navigation_simulator.ocean_observer.models.OceanCurrentModel import OceanCurrentModel


class OceanCurrentGP(OceanCurrentModel):
    """Wrapper around a Gaussian Process that handles ocean currents.
  This object models deviations from the forecast ("errors") using a Gaussian
  process over the 3-dimensional space (x, y, time).
  New measurements are integrated into the GP. Queries return the GP's
  prediction regarding particular 3D location's current in u, v format, plus
  the GP's confidence about that current.
  """

    def __init__(self, config_dict: Dict[str, Any]):
        """Constructor for the OceanCurrentGP.

        Args:
          config_dict: the config dictionary that setups the parameters for the Gaussian Processing
        """
        super().__init__()
        self.config_dict = config_dict
        self.life_span_observations_in_sec = config_dict.get("life_span_observations_in_sec", 24 * 3600)  # 24 hours.

        parameters_model = {}
        if "kernel" in self.config_dict:
            parameters_model["kernel"] = self.__get_kernel(self.config_dict["kernel"])
        if "sigma_noise_squared" in self.config_dict:
            parameters_model["alpha"] = self.config_dict["sigma_noise_squared"]
        if "optimizer" in self.config_dict:
            parameters_model["optimizer"] = self.config_dict["optimizer"]
        print("PARAMETERS_MODEL:", parameters_model)
        self.model = gaussian_process.GaussianProcessRegressor(**parameters_model)
        print(f"Gaussian Process created: {self.model}")

    def __get_kernel(self, dic_config: dict[str, Any]) -> Kernel:
        """Get the GP kernel based on the dictionary generated based on the Yaml file.
        Args:
            dic_config: Dictionary containing the hyper-parameters of the kernel that will be used by the GP.

        Returns:
            The created kernel. If the kernel specified is not supported, the constant kernel is returned
        """
        type_kernel = str(dic_config["type"])
        factor = dic_config.get("sigma_exp_squared", 1)
        params = dic_config.get("parameters", {})
        scales = dic_config.get("scaling", None)
        if scales is not None:
            params["length_scale"] = np.array([
                scales.get("longitude", 1), scales.get("latitude", 1), scales.get("time", 1)])
        if type_kernel.lower() == "rbf":
            return factor * gaussian_process.kernels.RBF(**params)
        if type_kernel.lower() == "matern":
            return factor * gaussian_process.kernels.Matern(**params)
        if type_kernel.lower() == "constantkernel":
            return factor * gaussian_process.kernels.ConstantKernel(**params)
        if type_kernel.lower() == "rationalquadratic":
            return factor * gaussian_process.kernels.RationalQuadratic(**params)
        if type_kernel.lower() == "expsinesquared":
            return factor * gaussian_process.kernels.ExpSineSquared(**params)
        # Not supported yet
        # if type_kernel.lower() == "sum":
        #     return self.__get_kernel(dic_config["kernel_1"]) + self.__get_kernel(dic_config["kernel_2"])
        # if type_kernel.lower() == "product":
        #     return self.__get_kernel(dic_config["kernel_1"]) + self.__get_kernel(dic_config["kernel_2"])

        print("No kernel specified in the yaml file. The constant kernel is used")
        return factor * gaussian_process.kernels.ConstantKernel()

    def fit(self) -> None:
        """Fit the Gaussian process using the observations(self.measurement_locations and self.measured_current_errors).
         Remove definitely the observations that are more than life_span_observations_in_sec older
         than the most recent one.
        """
        if not len(self.measurement_locations):
            print("no measurement_locations. Nothing to fit")
            return

        measurement_locations = np.array(self.measurement_locations, ndmin=2)
        errors = np.array(self.measured_current_errors, ndmin=2)

        if not np.all(measurement_locations[:, -1] == measurement_locations[0, -1]):
            most_recent_time = measurement_locations[:, -1].max()
            fresh_observations = (
                    np.abs(
                        measurement_locations[:, -1] - most_recent_time) < self.life_span_observations_in_sec)

            measurement_locations = measurement_locations[fresh_observations]
            errors = errors[fresh_observations]
            self.measurement_locations = list(measurement_locations)
            self.measured_current_errors = list(errors)

        # print("fitting:", len(measurement_locations), len(errors), measurement_locations, errors)
        # if len(measurement_locations) > 2:
        #     print("distance travelled:",
        #           np.array(((measurement_locations[1:] - measurement_locations[:-1]) ** 2)[:, :2].sum(
        #               axis=1) ** .5) * 111000, "meters")
        self.model.fit(measurement_locations, errors)

    def reset(self) -> None:
        """ Resetting the GP consist in removing all the observations."""
        super().reset()

    def get_predictions(self, locations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Compute the predictions for the given locations

        Args:
            locations: (N,3) ndarray that contains all the points we want to predict. Each point should be described by:
                       (lon in degree, lat in degree, time in datetime.datetime format).
        Returns:
            the predictions (N,3) and the std of these predictions (N,3).
        """
        locations[:, -1] = np.array(list(map(lambda x: x.timestamp(), locations[:, -1])))

        if not len(self.measurement_locations):
            means = np.zeros((locations.shape[0], 2))
            # TODO: define what should be the dev in that case if normalized then 1 is probably correct
            deviations = np.ones((locations.shape[0], 2))
            return means, deviations
        means, deviations = self.model.predict(locations, return_std=True)
        # Deviations are std.dev., convert to variance and normalize.
        # TODO: Check what the actual lower bound is supposed to
        # be. We can't have a 0 std.dev. due to noise. Currently it's something
        # like 0.07 from the GP, but that doesn't seem to match the Loon code.
        deviations = deviations ** 2 / self.config_dict.get("sigma_exp_squared", 1)

        return means, deviations
