from typing import NamedTuple

import numpy as np

import ocean_navigation_simulator.utils.units as units

# OceanCurrentVector contains the following elements:
#   u: Current magnitude along the x axis in meters per second.
#   v: Current magnitude along the y axis in meters per second.
# Note: x axis is parallel to latitude, and y axis is parallel to longitude.


class OceanCurrentVector(NamedTuple):
    """Describes the OceanCurrents at a given location."""

    u: units.Velocity
    v: units.Velocity

    def __post_init__(self):
        self._is_multi_agent = self.u.is_array

    def add(self, other: "OceanCurrentVector") -> "OceanCurrentVector":
        if not isinstance(other, OceanCurrentVector):
            raise NotImplementedError(f"Cannot add OceanCurrentVector with {type(other)}")
        return OceanCurrentVector(self.u + other.u, self.v + other.v)

    def subtract(self, other: "OceanCurrentVector") -> "OceanCurrentVector":
        if not isinstance(other, OceanCurrentVector):
            raise NotImplementedError(f"Cannot subtract OceanCurrentVector with {type(other)}")
        return OceanCurrentVector(self.u - other.u, self.v - other.v)

    def __array__(self) -> np.ndarray:
        # for multi_agent: outputs -> first row = u velocities, second row = v velocities
        return np.array([self.u, self.v]).squeeze()

    def __str__(self) -> str:
        return f"({self.u}, {self.v})"

    def __get_item__(self, item):
        return OceanCurrentVector(u=self.u[item], v=self.v[item])

    @staticmethod
    def from_numpy(arr):
        """
        Helper function to initialize a OceanCurrentVector based on numpy arraay.
        """
        #  for multi_agent first row -> u vectors, second row -> v vectors
        return OceanCurrentVector(u=units.Velocity(mps=arr[0]), v=units.Velocity(mps=arr[1]))
