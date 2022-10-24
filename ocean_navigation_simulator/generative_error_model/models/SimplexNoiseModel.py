# Vectorized and adapted version of:
# https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/env/wind_field.py
from ocean_navigation_simulator.utils import units

import dataclasses
import opensimplex
import numpy as np
import datetime
from typing import List


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


# OpenSimplex variance for a 3D noise field
OPENSIMPLEX_VARIANCE = 0.0979
# The noise's variance should match the global variance. Therefore, the
# noise is scaled.
NOISE_MAGNITUDE = np.sqrt(1.02/OPENSIMPLEX_VARIANCE)


class SimplexNoiseModel:
    """Noise model which based on simplex noise and parameters found through
    variogram analysis will generate noise at a point or over a volume"""

    def __init__(self, u_comp_harmonics: List[HarmonicParameters], v_comp_harmonics: List[HarmonicParameters]):
        self.noise_u = NoisyCurrentComponent(u_comp_harmonics)
        self.noise_v = NoisyCurrentComponent(v_comp_harmonics)

    def reset(self, rng) -> None:
        noise_u_rng, noise_v_rng = rng.choice(1894405231, size=2)
        noise_u_rng = np.random.default_rng(noise_u_rng)
        noise_v_rng = np.random.default_rng(noise_v_rng)
        self.noise_u.reset(noise_u_rng)
        self.noise_v.reset(noise_v_rng)

    def get_noise(self, x: np.ndarray, y: np.ndarray, elapsed_time: np.ndarray) -> np.ndarray:
        noise_u_value = self.noise_u.get_noise(x, y, elapsed_time)
        noise_v_value = self.noise_v.get_noise(x, y, elapsed_time)

        # reshape noise into x,y,z
        # noise_u_value = np.moveaxis(noise_u_value, [2, 1, 0], [0, 1, 2])
        # noise_v_value = np.moveaxis(noise_v_value, [2, 1, 0], [0, 1, 2])

        noise = np.array([noise_u_value, noise_v_value])
        noise = np.moveaxis(noise, [*range(1, len(noise.shape)), 0], list(range(0, len(noise.shape))))
        return noise


class NoisyCurrentComponent:
    """Computed from multiple NoisyCurrentHarmonics."""

    def __init__(self, comp_harmonics: List[HarmonicParameters]):

        harmonic_params = comp_harmonics

        # create list of harmonics
        self._harmonics = [
            NoisyCurrentHarmonic(params) for params in harmonic_params
        ]

    def reset(self, rng) -> None:
        """Create new rng for each harmonic from original rng."""
        num_harmonics = len(self._harmonics)
        harmonics_rngs_seeds = rng.choice(1894405231, size=num_harmonics)
        harmonics_rngs = [np.random.default_rng(seed) for seed in harmonics_rngs_seeds]

        for rng, harmonic in zip(harmonics_rngs, self._harmonics):
            harmonic.reset(rng)

    def get_noise(self, x: np.ndarray, y: np.ndarray, elapsed_time: np.ndarray) -> np.ndarray:
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
        # Note: Dont immediately multiply by sqrt of weight in loop because want to normalize weight
        weighted_noise /= total_weight
        # Note: need to scale in this way because actual weight is proportional to sill (harmonic weight)
        weighted_noise *= np.sqrt(total_weight / total_weight_squared)

        return weighted_noise


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

        random_translation = rng.uniform(size=(3,)) * 2.0 - 1.0
        self._offsets = SimplexOffset(*random_translation)

    def get_noise(self, x: np.ndarray, y: np.ndarray, elapsed_times: np.ndarray) -> np.ndarray:
        """Returns simplex noise for this harmonic at specific point in time and space."""

        if self._simplex_generator is None:
            raise ValueError("Must call reset before get_noise.")

        time_in_hours = np.array([elapsed_time.astype('timedelta64[h]')/np.timedelta64(1, 'h')
                                  for elapsed_time in elapsed_times])

        # magnitude is needed to scale the noise to a variance of 1.
        # The opensimplex method returns an array with the axis reversed wrt input order.
        return NOISE_MAGNITUDE * self._simplex_generator.noise3array(
            x / self.x_range + self._offsets.x,
            y / self.y_range + self._offsets.y,
            time_in_hours / self.time_range + self._offsets.time)


def test():
    # define the components instead of receiving them from OceanCurrentNoiseField
    u_comp = [
        HarmonicParameters(0.5, 702.5, 1407.3, 245.0),
        HarmonicParameters(0.5, 302.5, 1207.3, 187.0)
    ]
    v_comp = [
        HarmonicParameters(0.5, 702.5, 1407.3, 245.0),
        HarmonicParameters(0.5, 302.5, 1207.3, 187.0)
    ]

    noise = SimplexNoiseModel(u_comp, v_comp)
    noise.reset(np.random.default_rng(123))
    elapsed_times = np.array([np.timedelta64(5, 'h'), np.timedelta64(6, 'h'), np.timedelta64(20, 'h')])
    output = noise.get_noise(np.array([200, 300, 500, 550]), np.array([300, 200, 100, 200, 300]), elapsed_times)
    assert output.shape == (3, 5, 4, 2)


if __name__ == "__main__":
    test()
    print("All tests passed.")
