from typing import Tuple, Dict, Any

import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter

from ocean_navigation_simulator.ocean_observer.models.OceanCurrentModel import OceanCurrentModel


class OceanCurrentKF(OceanCurrentModel):
    def __init__(self, config_dict: Dict[str, Any]):
        """Constructor for the OceanCurrentKF.

        Args:
          config_dict: the config dictionary that setups the parameters for the Gaussian Processing
        """
        super().__init__()
        self.config_dict = config_dict
        self.life_span_observations_in_sec = config_dict.get(
            "life_span_observations_in_sec", 24 * 3600
        )  # 24 hours.

        self.model = KalmanFilter(dim_x=3, dim_z=2)
        self.model.x = np.eye(3)
        self.model.F = np.eye(3)
        self.model.H = np.array([[1, 1, 0], [1, 1, 0]])
        self.model.P *= 10
        self.model.R = 0.001
        self.model.R = np.array([[0.01, 0.001], [0.001, 0.01]])
        self.model.Q = Q_discrete_white_noise(dim=3, dt=1, var=2)
        print(f"Gaussian Process created: {self.model}")

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
                np.abs(measurement_locations[:, -1] - most_recent_time)
                < self.life_span_observations_in_sec
            )

            measurement_locations = measurement_locations[fresh_observations]
            errors = errors[fresh_observations]
            self.measurement_locations = list(measurement_locations)
            self.measured_current_errors = list(errors)

        self.model.fit(measurement_locations, errors)

    def reset(self) -> None:
        """Resetting the GP consist in removing all the observations."""
        super().reset()

    def get_predictions(self, locations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the predictions for the given locations

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
        deviations = deviations**2 / self.config_dict.get("sigma_exp_squared", 1)

        return means, deviations
