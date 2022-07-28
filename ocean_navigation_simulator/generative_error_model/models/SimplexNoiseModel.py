# source: https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/env/wind_field.py

"""
Generative model to generate realistic spatio-temporal noise for ocean currents based on
simplex noise.
"""

import dataclasses
import opensimplex
import numpy as np
import datetime

from ocean_navigation_simulator.generative_error_model.models.GenerativeModel import GenerativeModel
from ocean_navigation_simulator.utils import units


@dataclasses.dataclass(frozen=True)
class HarmonicParameters(object):
  """Parameters for noise harmonic."""
  weight: float
  x_range: float
  y_range: float
  time_range: float


@dataclasses.dataclass(frozen=True)
class SimplexOffset(object):
  """Defines a displacement from the base simplex grid."""
  x: float
  y: float
  time: float


@dataclasses.dataclass(frozen=True)
class CurrentVector(object):
    """Describes the wind at a given location."""
    u: units.Velocity
    v: units.Velocity

    def add(self, other: 'CurrentVector') -> 'CurrentVector':
        if not isinstance(other, CurrentVector):
            raise NotImplementedError(
                f'Cannot add CurrentVector with {type(other)}')
        return CurrentVector(self.u + other.u, self.v + other.v)

    def __str__(self) -> str:
        return f'({self.u}, {self.v})'


# weight, x, y, time.
_U_COMPONENT_HARMONICS = [
    HarmonicParameters(0.5, 702.5, 1407.3, 245.0),
    HarmonicParameters(0.5, 302.5, 1207.3, 187.0)
]

_V_COMPONENT_HARMONICS = [
    HarmonicParameters(0.5, 702.5, 1407.3, 245.0),
    HarmonicParameters(0.5, 302.5, 1207.3, 187.0)
]


OPENSIMPLEX_VARIANCE = 0.0569
# The noise's variance should match the global variance. Therefore, the
# noise is scaled.
NOISE_MAGNITUDE = np.sqrt(1.02/OPENSIMPLEX_VARIANCE)


class SimplexNoiseModel(GenerativeModel):
    """Noise model which based on simplex noise and parameters found through
    variogram analysis will generate noise at a point or over a volume"""

    def __init__(self):
        self.noise_u = NoisyCurrentComponent(component="u")
        self.noise_v = NoisyCurrentComponent(component="v")

    def reset(self, rng) -> None:
        noise_u_rng, noise_v_rng = rng.choice(1894405231, size=2)
        noise_u_rng = np.random.default_rng(noise_u_rng)
        noise_v_rng = np.random.default_rng(noise_v_rng)
        self.noise_u.reset(noise_u_rng)
        self.noise_v.reset(noise_v_rng)

    def get_noise(self, x: units.Distance, y: units.Distance,
        elapsed_time: datetime.timedelta) -> CurrentVector:
        noise_u_value = self.noise_u.get_noise(x, y, elapsed_time)
        noise_v_value = self.noise_v.get_noise(x, y, elapsed_time)

        return CurrentVector(
            units.Velocity(meters_per_second=noise_u_value),
            units.Velocity(meters_per_second=noise_v_value)
        )


class NoisyCurrentHarmonic:
    """Sub-component of NoisyCurrentComponent. Multiple weighted NoisyCurrentHarmonics
    make up a NoisyCurrentComponent."""

    def __init__(self, params: HarmonicParameters):
        self._simplex_generator = None
        self._offsets = None

        # unpacking harmonic components
        self.weight = params.weight
        self.x_range = params.x_range
        self.y_range = params.y_range
        self.time_range = params.time_range

    def reset(self, rng) -> None:
        """Resets harmonic's noise."""
        current_seed = rng.choice(1894405231)
        self._simplex_generator = opensimplex.OpenSimplex(seed=current_seed)

        # OpenSimplex always return zero at origin so for a given
        # generated noise a random offset from the origin is used

        random_translation = rng.uniform(low=0, high=1894405231, size=(3,)) * 2.0 - 1.0
        self._offsets = SimplexOffset(*random_translation)

    def get_noise(self, x: units.Distance, y: units.Distance, elapsed_time: datetime.timedelta) -> float:
        """Returns simplex noise for this harmonic at specific point in time and space."""

        if self._simplex_generator is None:
            raise ValueError("Must call reset before get_noise.")

        time_in_hours = units.timedelta_to_hours(elapsed_time)

        return NOISE_MAGNITUDE * self._simplex_generator.noise3(
            x.km / self.x_range + self._offsets.x,
            y.km / self.y_range + self._offsets.y,
            time_in_hours / self.time_range + self._offsets.time)

class NoisyCurrentComponent:
    """Computed from multiple NoisyCurrentHarmonics."""

    def __init__(self, component: str):
        """
        Args:
            components: "u" or "v".
        """

        if component == "u":
            harmonic_params = _U_COMPONENT_HARMONICS
        elif component == "v":
            harmonic_params = _V_COMPONENT_HARMONICS
        else:
            raise RuntimeError((f"Invalid current component: {component}"))

        # create list of harmonics
        self._harmonics = [
            NoisyCurrentHarmonic(params) for params in harmonic_params
        ]

    def reset(self, rng) -> None:
        num_harmonics = len(self._harmonics)
        harmonics_rngs_seeds = rng.choice(1894405231, size=num_harmonics)
        harmonics_rngs = [np.random.default_rng(seed) for seed in harmonics_rngs_seeds]

        for rng, harmonic in zip(harmonics_rngs, self._harmonics):
            harmonic.reset(rng)

    def get_noise(self, x:units.Distance, y: units.Distance, elapsed_time: datetime.timedelta) -> float:
        """Returns simplex noise at defined location (in space and time), for specified components."""

        weighted_noise = 0.0
        total_weight = 0.0
        total_weight_squared = 0.0

        # Sum noise from all harmonics and weight
        for harmonic in self._harmonics:
            noise = harmonic.get_noise(x, y, elapsed_time)
            weighted_noise += noise * harmonic.weight
            total_weight += harmonic.weight
            total_weight_squared += harmonic.weight**2

        # Scale to recover the desired empirical variance
        weighted_noise /= total_weight
        weighted_noise *= np.sqrt(total_weight / total_weight_squared)

        return weighted_noise


#### test funcs ####

def test_NoisyCurrentHarmonic():

    rng = np.random.default_rng(2022)

    harmonic = NoisyCurrentHarmonic(_U_COMPONENT_HARMONICS[0])
    harmonic.reset(rng)
    harmonic_noise1 =  harmonic.get_noise(
        units.Distance(km=200),
        units.Distance(km=300),
        datetime.timedelta(days=5))

    harmonic_noise2 =  harmonic.get_noise(
        units.Distance(km=200),
        units.Distance(km=300),
        datetime.timedelta(days=5))

    assert(harmonic_noise1 == harmonic_noise2), "NoisyCurrentHarmonic test failed!"

def test_NoisyCurrentComponent():

    rng = np.random.default_rng(2022)

    component = NoisyCurrentComponent("u")
    component.reset(rng)
    return component.get_noise(units.Distance(km=200), units.Distance(km=300), datetime.timedelta(days=5))

def test_SimplexNoiseModel():

    rng = np.random.default_rng(2022)

    noise_model = SimplexNoiseModel()
    noise_model.reset(rng)
    velocity1 = noise_model.get_noise(
        units.Distance(km=200),
        units.Distance(km=300),
        datetime.timedelta(days=5))

    velocity2 = noise_model.get_noise(
        units.Distance(km=200),
        units.Distance(km=300),
        datetime.timedelta(days=5))

    assert(velocity1 == velocity2), "SimplexNoiseModel test failed!"


if __name__ == "__main__":
    test_NoisyCurrentHarmonic()
    test_SimplexNoiseModel()
    print("All tests passed.")