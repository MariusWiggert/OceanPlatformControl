import dataclasses
import datetime
import math
from typing import List

import numpy as np

from ocean_navigation_simulator.utils import units


@dataclasses.dataclass
class SpatialPoint:
    """A dataclass containing variables that define the spatial position.

    Attributes:
      lon: The latitude position in degree
      lat: The longitude position in degree
    """

    lon: units.Distance
    lat: units.Distance

    def distance(self, other) -> units.Distance:
        return units.Distance(
            deg=math.sqrt((self.lat.deg - other.lat.deg) ** 2 + (self.lon.deg - other.lon.deg) ** 2)
        )

    def haversine(self, other) -> units.Distance:
        """
        Calculate the great circle distance in degrees between two points
        on the earth (specified in decimal degrees)
        Taken from: https://stackoverflow.com/a/4913653
        """
        return units.Distance(
            rad=units.haversine_rad_from_deg(
                self.lon.deg, self.lat.deg, other.lon.deg, other.lat.deg
            )
        )

    def __array__(self):
        return np.array([self.lon.deg, self.lat.deg])

    def __len__(self):
        return self.__array__().shape[0]

    def __getitem__(self, item):
        return self.__array__()[item]

    def __repr__(self):
        return f"[{self.lon.deg:5f}°,{self.lat.deg:.5f}°]"


@dataclasses.dataclass
class SpatioTemporalPoint:
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

    def distance(self, other) -> units.Distance:
        return self.to_spatial_point().distance(other)

    def haversine(self, other) -> units.Distance:
        return self.to_spatial_point().haversine(other)

    def to_spatial_point(self) -> SpatialPoint:
        """Helper function to just extract the spatial point."""
        return SpatialPoint(lon=self.lon, lat=self.lat)

    def to_spatio_temporal_casadi_input(self) -> List[float]:
        """Helper function to produce a list [posix_time, lat, lon] to feed into casadi."""
        return [self.date_time.timestamp(), self.lat.deg, self.lon.deg]

    def __repr__(self):
        return f"[{self.lon.deg:5f}°,{self.lat.deg:.5f}°,{self.date_time.strftime('%Y-%m-%d %H:%M:%S')}]"


@dataclasses.dataclass
class PlatformState:
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
        return np.array(
            [
                self.lon.deg,
                self.lat.deg,
                self.date_time.timestamp(),
                self.battery_charge.joule,
                self.seaweed_mass.kg,
            ]
        )

    def __len__(self):
        return self.__array__().shape[0]

    def __getitem__(self, item):
        return self.__array__()[item]

    def distance(self, other) -> units.Distance:
        return self.to_spatial_point().distance(other)

    def haversine(self, other) -> units.Distance:
        return self.to_spatial_point().haversine(other)

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
            seaweed_mass=units.Mass(kg=numpy_array[4]),
        )

    @staticmethod
    def from_spatio_temporal_point(point: SpatioTemporalPoint):
        return PlatformState(lon=point.lon, lat=point.lat, date_time=point.date_time)

    def to_spatial_point(self) -> SpatialPoint:
        """Helper function to just extract the spatial point."""
        return SpatialPoint(lon=self.lon, lat=self.lat)

    def to_spatio_temporal_point(self) -> SpatioTemporalPoint:
        """Helper function to just extract the spatio-temporal point."""
        return SpatioTemporalPoint(lon=self.lon, lat=self.lat, date_time=self.date_time)

    def to_spatio_temporal_casadi_input(self) -> List[float]:
        """Helper function to produce a list [posix_time, lat, lon] to feed into casadi."""
        return [self.date_time.timestamp(), self.lat.deg, self.lon.deg]

    def __repr__(self):
        return "Platform State[lon: {x} deg, lat: {y} deg, date_time: {t}, battery_charge: {b} Joule, seaweed_mass: {m} kg]".format(
            x=self.lon.deg,
            y=self.lat.deg,
            b=self.battery_charge.joule,
            m=self.seaweed_mass.kg,
            t=self.date_time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def distance(self, other) -> float:
        return self.to_spatial_point().distance(other)
