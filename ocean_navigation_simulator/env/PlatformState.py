import dataclasses
import datetime
from ocean_navigation_simulator.env.utils import units
from typing import List


@dataclasses.dataclass
class SpatialPoint:
    """A dataclass containing variables that define the spatial position.

      Attributes:
        lon: The latitude position in degree
        lat: The longitude position in degree
      """
    lon: units.Distance
    lat: units.Distance


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
    date_time: datetime.datetime = datetime.datetime.now(tz=datetime.timezone.utc)


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
    battery_charge: units.Energy = units.Energy(watt_hours=100)
    seaweed_mass: units.Mass = units.Mass(kg=100)
    date_time: datetime.datetime = datetime.datetime.now(tz=datetime.timezone.utc)

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
        return 'Platform State: lon: {x} deg, lat: {y} deg, battery_charge: {b} Joule, seaweed_mass: {m} kg, ' \
               'date_time: {t}'.format(
            x=self.lon.deg, y=self.lat.deg, b=self.battery_charge.joule, m=self.seaweed_mass.kg, t=self.date_time
        )






