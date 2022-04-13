import dataclasses
import datetime
from ocean_navigation_simulator.env.utils import units


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