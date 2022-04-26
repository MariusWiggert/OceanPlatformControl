import dataclasses
import datetime
from ocean_navigation_simulator.env.utils import units
from typing import List
import numpy as np

@dataclasses.dataclass
class SpatialPoint():
    """A dataclass containing variables that define the spatial position.

      Attributes:
        lon: The latitude position in degree
        lat: The longitude position in degree
      """
    lon: units.Distance
    lat: units.Distance

    def __array__(self):
        return np.array([self.lon.deg, self.lat.deg])

    def __len__(self):
        return self.__array__().shape[0]

    def __getitem__(self, item):
        return self.__array__()[item]


@dataclasses.dataclass
class SpatioTemporalPoint():
    # TODO: implement nice way to transform a list of those to numpy and back: https://kplauritzen.dk/2021/08/11/convert-dataclasss-np-array.html
    """A dataclass containing SpatioTemporalPoint variables..

      Attributes:
        lon: The latitude position in degree
        lat: The longitude position in degree
        date_time: The current time.
      """
    lon: units.Distance
    lat: units.Distance
    date_time: datetime.datetime

    def __array__(self):
        return np.array([self.lon.deg, self.lat.deg, self.date_time.timestamp()])

    def __len__(self):
        return self.__array__().shape[0]

    def __getitem__(self, item):
        return self.__array__()[item]

    def to_spatial_point(self) -> SpatialPoint:
        """Helper function to just extract the spatial point."""
        return SpatialPoint(lon=self.lon, lat=self.lat)

@dataclasses.dataclass
class PlatformState():
    """A dataclass containing variables relevant to the platform state.

      Attributes:
        lon: The latitude position in degree
        lat: The longitude position in degree
        battery_charge: The amount of energy stored on the batteries.
        seaweed_mass: The amount of seaweed biomass on the system.
        date_time: The current time.
      """
    lon: units.Distance
    lat: units.Distance
    date_time: datetime.datetime = datetime.datetime.now(tz=datetime.timezone.utc)
    battery_charge: units.Energy = units.Energy(joule=100)
    seaweed_mass: units.Mass = units.Mass(kg=100)

    def __array__(self):
        return np.array([self.lon.deg, self.lat.deg, self.date_time.timestamp(), self.battery_charge.joule, self.seaweed_mass.kg])

    def __len__(self):
        return self.__array__().shape[0]

    def __getitem__(self, item):
        return self.__array__()[item]

    @staticmethod
    def from_numpy(numpy_array):
        """Helper function to initialize a PlatformState based on numpy arraay.
        Args:
            numpy_array
        Returns:
            PlatformAction object
        """
        return PlatformState(
            lon=units.Distance(deg=numpy_array[0]),
            lat=units.Distance(deg=numpy_array[1]),
            date_time=datetime.datetime.fromtimestamp(numpy_array[2], tz=datetime.timezone.utc),
            battery_charge=units.Energy(joule=numpy_array[3]),
            seaweed_mass=units.Mass(kg=numpy_array[4])
        )

    def to_spatial_point(self) -> SpatialPoint:
        """Helper function to just extract the spatial point."""
        return SpatialPoint(lon=self.lon, lat=self.lat)

    def to_spatio_temporal_point(self) -> SpatioTemporalPoint:
        """Helper function to just extract the spatio-temporal point."""
        return SpatioTemporalPoint(lon=self.lon, lat=self.lat, date_time=self.date_time)

    def to_spatio_temporal_casadi_input(self) -> List[float]:
        """Helper function to produce a list [posix_time, lat, lon] to feed into casadi."""
        return [self.date_time.timestamp(), self.lat.deg, self.lon.deg]






