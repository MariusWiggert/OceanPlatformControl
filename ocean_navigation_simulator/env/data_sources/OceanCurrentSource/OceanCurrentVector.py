from typing import List, NamedTuple, Sequence, Optional
import ocean_navigation_simulator.env.utils.units as units

# OceanCurrentVector contains the following elements:
#   u: Current magnitude along the x axis in meters per second.
#   v: Current magnitude along the y axis in meters per second.
# Note: x axis is parallel to latitude, and y axis is parallel to longitude.
class OceanCurrentVector(NamedTuple):
    """Describes the OceanCurrents at a given location."""
    u: units.Velocity
    v: units.Velocity

    def add(self, other: 'OceanCurrentVector') -> 'OceanCurrentVector':
        if not isinstance(other, OceanCurrentVector):
            raise NotImplementedError(
                f'Cannot add OceanCurrentVector with {type(other)}')
        return OceanCurrentVector(self.u + other.u, self.v + other.v)

    def subtract(self, other: 'OceanCurrentVector') -> 'OceanCurrentVector':
        if not isinstance(other, OceanCurrentVector):
            raise NotImplementedError(
                f'Cannot add OceanCurrentVector with {type(other)}')
        return OceanCurrentVector(self.u - other.u, self.v - other.v)

    def __str__(self) -> str:
        return f'({self.u}, {self.v})'