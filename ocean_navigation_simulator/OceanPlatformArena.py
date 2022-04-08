"""A Ocean Platform Arena.
A Ocean arena contains the logic for navigating a platform in the ocean.
"""

import abc
import datetime as dt
import math
import time
from typing import Callable, Optional, Union

from ocean_navigation_simulator.data_sources.OceanCurrentFields import OceanCurrentField

# from balloon_learning_environment.env import features
# from balloon_learning_environment.env import simulator_data
# from balloon_learning_environment.env import wind_field
# from balloon_learning_environment.env.balloon import balloon
# from balloon_learning_environment.env.balloon import control
# from balloon_learning_environment.env.balloon import stable_init
# from balloon_learning_environment.env.balloon import standard_atmosphere
# from balloon_learning_environment.utils import constants
# from balloon_learning_environment.utils import sampling
# from balloon_learning_environment.utils import units
import jax
import jax.numpy as jnp
import numpy as np


class OceanPlatformArenaInterface(abc.ABC):
    """An interface for a Ocean Platform Arena.
  The ocean platform arena is the "simulator" for navigating platforms in the ocean.
  As such, and child class should encapsulate all functionality and data
  involved in navigating on the ocean, but not the controller.
  (which is in the Controller class).
  """

    @abc.abstractmethod
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Resets the arena.
    Args:
      seed: An optional seed for resetting the arena.
    Returns:
      The first observation from the newly reset simulator as a numpy array.
    """

    @abc.abstractmethod
    def step(self, action: control.AltitudeControlCommand) -> np.ndarray:
        """Steps the simulator.
    Args:
      action: The ocean platform control to apply.
    Returns:
      The observation from the simulator as a numpy array.
    """

    @abc.abstractmethod
    def get_simulator_state(self) -> simulator_data.SimulatorState:
        """Gets the current simulator state.
    This should return the full simulator state so that it can be used for
    checkpointing.
    Returns:
      The simulator state.
    """

    @abc.abstractmethod
    def set_simulator_state(self,
                            new_state: simulator_data.SimulatorState) -> None:
        """Sets the simulator state.
    This should fully restore the simulator state so that it can restore
    from a checkpoint.
    Args:
      new_state: The state to set the simulator to.
    """

    @abc.abstractmethod
    def get_platform_state(self) -> platform.PlatformState:
        """Gets the platform state.
    Returns:
      The current platform state.
    """

    @abc.abstractmethod
    def set_platform_state(self, new_state: platform.PlatformState) -> None:
        """Sets the platform state.
    Args:
      new_state: The state to set the platform to.
    """

    @abc.abstractmethod
    def get_measurements(self) -> simulator_data.SimulatorObservation:
        """Gets measurements from the arena.
    This is what a controller may feasibly use to control the platform.
    The feature processing happens in the gym environment.
    Returns:
      (Noisy) sensor readings of the current state.
    """


class OceanPlatformArena(OceanPlatformArenaInterface):
    """A OceanPlatformArena in which an ocean platform moves through a current field."""

    def __init__(self,
                 # TODO: where do we do the reset? I guess for us reset mostly would mean new start and goal position?
                 # Maybe we have the problem object in here? Then the gym environment has a list of problems as the training distribution and feeds them in here via resets?
                 # TODO: not sure what that should be for us, decide where to put the feature constructor
                 # feature_constructor_factory: Callable[
                 #     [wind_field.WindField, standard_atmosphere.Atmosphere],
                 #     features.FeatureConstructor],
                 ocean_field_instance: OceanCurrentField,
                 solar_field_instance: OceanCurrentField,
                 growth_field_instance: OceanCurrentField,
                 seed: Optional[int] = None):
        """OceanPlatformArena constructor.
    Args:
      feature_constructor_factory: A factory that when called returns an
        object that constructs feature vectors from observations. The factory
        takes a wind field (WindField) and an initial observation from the
        simulator (SimulatorObservation).
      ocean_field_instance: A OceanCurrentField to use in the simulation (for true currents and forecasts)
      seed: An optional seed for the arena. If it is not specified, it will be
        seeded based on the system time.
    """
        self._ocean_current_field = ocean_field_instance
        self._step_duration = constants.AGENT_TIME_STEP
        self._platform: platform.Platform  # Initialized in reset.

        # We call reset here to ensure the arena can always be run without
        # an explicit call to reset. However, the preferred way to run the
        # arena is to call reset immediately to return the intiial observation.
        self.reset(seed)

    def reset(self, seed: Union[int, jnp.ndarray, None] = None) -> np.ndarray:
        if isinstance(seed, int):
            self._rng = jax.random.PRNGKey(seed)
        elif isinstance(seed, (np.ndarray, jnp.ndarray)):
            self._rng = seed
        else:
            # Seed with time in microseconds
            self._rng = jax.random.PRNGKey(int(time.time() * 1e6))

        self._rng, atmosphere_key, time_key = jax.random.split(self._rng, 3)
        self._atmosphere.reset(atmosphere_key)
        start_date_time = sampling.sample_time(time_key)
        self._balloon = self._initialize_balloon(start_date_time)
        assert self._balloon.state.status == balloon.BalloonStatus.OK

        self._rng, wind_field_key = jax.random.split(self._rng, 2)
        self._wind_field.reset(wind_field_key, start_date_time)

        self.feature_constructor = self._feature_constructor_factory(
            self._wind_field, self._atmosphere)
        self.feature_constructor.observe(self.get_measurements())
        return self.feature_constructor.get_features()

    def step(self, action: control.AltitudeControlCommand) -> np.ndarray:
        """Simulates the effects of choosing the given action in the system.
    Args:
      action: The action to take in the simulator.
    Returns:
      A feature vector (numpy array) constructed by the feature constructor.
    """
        # Determine the wind at the balloon's location.
        wind_vector = self._get_wind_ground_truth_at_balloon()

        # Simulate the balloon dynamics in the wind field seconds.
        self._balloon.simulate_step(wind_vector, self._atmosphere, action,
                                    self._step_duration)

        # At the end of the cycle, make a measurement, and construct features.
        self.feature_constructor.observe(self.get_measurements())
        return self.feature_constructor.get_features()

    def get_simulator_state(self) -> simulator_data.SimulatorState:
        return simulator_data.SimulatorState(self.get_balloon_state(),
                                             self._wind_field,
                                             self._atmosphere)

    def set_simulator_state(self,
                            new_state: simulator_data.SimulatorState) -> None:
        # TODO(joshgreaves): Restore the state of the feature constructor.
        self.set_balloon_state(new_state.balloon_state)
        self._wind_field = new_state.wind_field
        self._atmosphere = new_state.atmosphere

    def get_balloon_state(self) -> balloon.BalloonState:
        return self._balloon.state

    def set_balloon_state(self, new_state: balloon.BalloonState) -> None:
        self._balloon.state = new_state

    def get_measurements(self) -> simulator_data.SimulatorObservation:
        # TODO(joshgreaves): Add noise to observations
        return simulator_data.SimulatorObservation(
            balloon_observation=self.get_balloon_state(),
            wind_at_balloon=self._get_wind_ground_truth_at_balloon())

    def _initialize_balloon(self,
                            start_date_time: dt.datetime) -> balloon.Balloon:
        """Initializes a balloon.
    Initializes a balloon within 200km of the target. The balloon's distance
    from the target is sampled from a beta distribution, while the direction
    (angle) is sampled uniformly. Its pressure is also sampled uniformly
    from all valid pressures.
    Args:
      start_date_time: The starting date and time.
    Returns:
      A new balloon object.
    """
        self._rng, *keys = jax.random.split(self._rng, num=6)

        # Note: Balloon units are in Pa.
        # Sample the starting distance using a beta distribution, within 200km.
        radius = jax.random.beta(keys[0], self._alpha, self._beta).item()
        radius = units.Distance(km=200.0 * radius)
        theta = jax.random.uniform(keys[1], (), minval=0.0, maxval=2.0 * jnp.pi)

        x = math.cos(theta) * radius
        y = math.sin(theta) * radius
        # TODO(bellemare): Latitude in the tropics, otherwise around the world.
        # Does longitude actually affect anything?
        # TODO(joshgreaves): sample_location only samples between -10 and 10 lat.
        latlng = sampling.sample_location(keys[2])

        pressure = sampling.sample_pressure(keys[3], self._atmosphere)
        upwelling_infrared = sampling.sample_upwelling_infrared(keys[4])
        b = balloon.Balloon(
            balloon.BalloonState(
                center_latlng=latlng,
                x=x,
                y=y,
                pressure=pressure,
                date_time=start_date_time,
                upwelling_infrared=upwelling_infrared))
        stable_init.cold_start_to_stable_params(b.state, self._atmosphere)
        return b

    def _get_wind_ground_truth_at_balloon(self) -> wind_field.WindVector:
        """Returns the wind vector at the balloon's current location."""
        return self._wind_field.get_ground_truth(self._balloon.state.x,
                                                 self._balloon.state.y,
                                                 self._balloon.state.pressure,
                                                 self._balloon.state.time_elapsed)
