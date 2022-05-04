"""Class for summarizing ocean currents observations and forecast.
We use a Gaussian Process to integrate wind observations to a basic forecast.
This lets us query any point (x, y, t) in the ocean current field for its value,
as well as the model confidence's in this value.
---- Open issues
* The forecast is not used

source: https://github.com/google/balloon-learning-environment/blob/cdb2e582f2b03c41f037bf76142d31611f5e0316/balloon_learning_environment/env/wind_gp.py
"""

import datetime as dt
from typing import Tuple

import numpy as np
from sklearn import gaussian_process
from typing import List

from ocean_navigation_simulator.env.PlatformState import SpatioTemporalPoint
from ocean_navigation_simulator.env.data_sources.OceanCurrentField import OceanCurrentField
from ocean_navigation_simulator.env.data_sources.OceanCurrentSource import OceanCurrentSource
from ocean_navigation_simulator.env.utils import units
from ocean_navigation_simulator.env.utils.units import Distance

_LATITUDE_SCALING = 1 # [m]
_LONGITUDE_SCALING = 1 # [m]
_TIME_SCALING = 50  # [seconds]

_SIGMA_EXP_SQUARED = 3.6 ** 2
_SIGMA_NOISE_SQUARED = 0.05


class OceanCurrentGP(object):
    """Wrapper around a Gaussian Process that handles ocean currents.
  This object models deviations from the forecast ("errors") using a Gaussian
  process over the 3-dimensional space (x, y, time).
  New measurements are integrated into the GP. Queries return the GP's
  prediction regarding particular 3D location's current in u, v format, plus
  the GP's confidence about that current.
  """

    def __init__(self, forecast: OceanCurrentField) -> None:
        """Constructor for the OceanCurrentGP.
    TODO(bellemare): Currently a forecast is required. This simplifies the
    code somewhat. Whether we keep this depends on a design choice: is a new
    environment built up and torn down per episode, or do we instead use
    reset() functions to reuse objects?
    Args:
      forecast: the forecast ocean current field.
    """
        self.ocean_current_forecast = None
        self.measurement_locations = None
        self.error_values = None
        # TODO: VERIFY time
        self.time_horizon = 24 * 3600  # 24 hours.

        # The OceanCurrentGP kernel is a Matern kernel.
        # This rescales the inputs (or equivalently, the distance) by the given
        # scaling factors.
        length_scale = np.array([
            _LATITUDE_SCALING, _LONGITUDE_SCALING, _TIME_SCALING])
        self.kernel = _SIGMA_EXP_SQUARED * gaussian_process.kernels.Matern(
            length_scale=length_scale,
            length_scale_bounds='fixed', nu=0.5)

        self.model = gaussian_process.GaussianProcessRegressor(
            kernel=self.kernel,  # Matern kernel.
            alpha=_SIGMA_NOISE_SQUARED,  # Add a term to the diagonal of the kernel.
            optimizer=None,  # No optimization.
        )
        self.reset(forecast=forecast)

    def reset(self, forecast: OceanCurrentField) -> None:
        """Resets the the OceanCurrentGP, effectively erasing previous measurements.
    Args:
      forecast: a 3D forecast for the entire ocean current field.
    """
        # Erase measurements. Since scikit's GP is all runtime, this is just
        # clearing the list of points.
        self.measurement_locations = []
        self.error_values = []

        # TODO(bellemare): This may change types, as OceanCurrentField is the 'true'
        # ocean field and we may instead want the 'mean' ocean current field (pre-OC noise).
        self.ocean_current_forecast = forecast

    def observe(self, x: float, y: float,
                time: dt.datetime,
                error: OceanCurrentSource.OceanCurrentVector) -> None:
        """Adds the given measurement to the Gaussian Process.
    Args:
      x: latitude coordinate
      y: longitude coordinate
      time: datetime of the measurement.
      error: The ocean current error at the location.
    """
        location = np.array([x, y, time])
        x,y = Distance(deg=x), Distance(deg=y)
        #forecast = self.ocean_current_forecast.get_forecast(SpatioTemporalPoint(lon=x, lat=y, date_time=time))
        #error = np.array([(measurement.u - forecast.u),
        #                  (measurement.v - forecast.v)])
        self.measurement_locations.append(location)
        self.error_values.append(error)

    def query(self, x: float, y: float, t_0: dt.datetime) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the GP's ocean current prediction at the given location.
    Args:
      x: the x query coordinate = longitude.
      y: the y query coordinate = latitude.
      t_0: the time at which to query the GP.
    Returns:
      u: the mean ocean current direction in x.
      v: the mean ocean current direction in y.
      confidence: the GP's confidence in these values.
    """
        query_as_array = np.array(
            [[x, y, t_0]])
        outputs = self.query_batch(query_as_array)
        # Remove the batch dimension.
        return outputs[0][0], outputs[1][0]

    def query_batch(self, locations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the GP's ocean currents prediction for a batch of queries.
    Args:
      locations: a N x 3 dimensional array of queries. Each row contains the
        (x, y, t) coordinates for one query.
        The 't' argument should be in datetime format.
    Returns:
      means: a N x 2 dimensional array. Each row contains the mean ocean currents
        direction (x and y).
      confidence: a N-dim vector. Each element contains the model's uncertainty.
    Raises:
      RuntimeError: if no forecast was previously given.
    """
        # TODO(Killian): not sure if necessary, Why are the deviations 0 if no measurements
        # Set up data for the GP.
        # TODO(bellemare): Clearly wasteful if performing multiple queries per
        # observation. Should cache. Premature optimization is the root, etc.
        if not self.measurement_locations:
            means = np.zeros((locations.shape[0], 2))
            deviations = np.zeros(locations.shape[0])
        else:
            if len(self.measurement_locations) == 1:
                inputs = np.expand_dims(self.measurement_locations[0], axis=0)
                targets = np.expand_dims(self.error_values[0], axis=0)
            else:
                inputs = np.vstack(self.measurement_locations)
                targets = np.vstack(self.error_values)
            # Drop any observations that are more than N hours old. This speeds up
            # computation. Only if all queries have the same time.
            # TODO(bellemare): A slightly more efficient alternative is to drop the
            # data permanently, but this method has the advantage of supporting
            # queries into the past.
            if np.all(locations[:, -1] == locations[0, -1]):
                current_time = locations[0, -1]
                fresh_observations = (
                        np.abs(
                            list(map(lambda x: x.total_seconds(), inputs[:, -1] - np.array(current_time)))
                        ) < self.time_horizon
                )
                print("fresh_observations:",np.sum(fresh_observations))
                inputs = inputs[fresh_observations]
                targets = targets[fresh_observations]
            #Use a timestamp instead of datetime format
            inputs[:, -1] = np.array(list(map(lambda x: x.timestamp(), inputs[:, -1])))
            copy_loc = np.array(locations)
            copy_loc[:, -1] = np.array(list(map(lambda x: x.timestamp(), copy_loc[:, -1])))
            # We fit here the [x, y, t] coordinates with the error between forecasts and hindcasts
            self.model.fit(inputs, targets)
            # Output should be a N x 2 set of predictions about local measurements,
            # and a N-sized vector of standard deviations.
            means, deviations = self.model.predict(copy_loc, return_std=True)
            # Deviations are std.dev., convert to variance and normalize.
            # TODO(bellemare): Ask what the actual lower bound is supposed to
            # be. We can't have a 0 std.dev. due to noise. Currently it's something
            # like 0.07 from the GP, but that doesn't seem to match the Loon code.
            deviations = deviations ** 2 / _SIGMA_EXP_SQUARED

            # TODO(bellemare): Sal says this needs normalizing so that the lower bound
            # is really zero.

        # Verify that the shape of mean should be 1 (as there is no pressure dimension)
        assert len(means.shape) == 2, means.shape[1] == 2

        self._add_forecast_to_prediction(locations, means)
        return means, deviations

    def fitting_GP(self):
        # TODO(Killian): not sure if necessary, Why are the deviations 0 if no measurements
        # Set up data for the GP.
        # TODO(bellemare): Clearly wasteful if performing multiple queries per
        # observation. Should cache. Premature optimization is the root, etc.
        if not self.measurement_locations:
            print("no measurement_locations. Nothing to fit")
        else:
            if len(self.measurement_locations) == 1:
                # Needed because hstack will leave the dims unchanged with a single
                # row.
                inputs = np.expand_dims(self.measurement_locations[0], axis=0)
                targets = np.expand_dims(self.error_values[0], axis=0)
            else:
                inputs = np.vstack(self.measurement_locations)
                targets = np.vstack(self.error_values)

            # Use a timestamp instead of datetime format
            inputs[:, -1] = np.array(list(map(lambda x: x.timestamp(), inputs[:, -1])))
            # We fit here the [x, y, t] coordinates with the error between forecasts and hindcasts
            self.model.fit(inputs, targets)
            #print("values fitted:", len(inputs), inputs, targets)

    def query_locations(self, locations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Convert the date to a timestamp
        locations[:, -1] = np.array(list(map(lambda x: x.timestamp(), locations[:, -1])))
        means, deviations = self.model.predict(locations, return_std=True)
        # reshape the predictions
        return means, deviations

    def _add_forecast_to_prediction(
            self, locations: np.ndarray, means: np.ndarray) -> None:
        """Adds the forecast back to the error prediction.
    The OceanCurrentGP predicts the error from the forecasts. When that is done, we
    need to recombine it with the forecast to obtain the actual prediction.
    The 'means' vector is modified in-place.
    Args:
      locations: N x 3 array of locations at which predictions have been made.
      means: 2D array of predicted deviations from the forecast.
    """
        # This checks that all x, y, and time are the same in each row.
        '''assert (locations[1:, [0, 1, 3]] == locations[0, [0, 1, 3]]).all()
    forecasts = self.wind_forecast.get_forecast_column(
        units.Distance(m=locations[0, 0]),
        units.Distance(m=locations[0, 1]),
        locations[:, 2],
        dt.timedelta(seconds=locations[0, 3]))
    for index, forecast in enumerate(forecasts):
      means[index][0] += forecast.u.meters_per_second
      means[index][1] += forecast.v.meters_per_second
    '''
        # In our case we don't have any third dimension like with the pressure
        # TODO(Killian): verify it is working
        # Should not be necessary as no pressure dimension
            # assert (locations[1:, [0,1,2]] == locations[0,[0,1,2]]).all() -> Means has
        #loc_date = dt.datetime.fromtimestamp(locations[0, 2])
        forecast = self.ocean_current_forecast.get_forecast(locations[0, 0:2], locations[0, 2])
        means[0, 0] += forecast[0]
        means[0, 1] += forecast[1]
