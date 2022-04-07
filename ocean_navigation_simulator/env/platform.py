"""Simulator for a seaweed platform.

Typical usage:

  current_field = ...
  platform = Platform()
  for _ in range(horizon / stride):
    command = ...
    platform.simulate_step(current_field, command, stride)
    print(platform.x, platform.y)
"""

import dataclasses

from ocean_navigation_simulator.utils import units
import datetime as dt

@dataclasses.dataclass
class PlatformState(object):
    """A dataclass containing variables relevant to the platform state.

      Attributes:
        date_time: The current time.
        time_elapsed: The time elapsed in simulation from the time the object was
            initialized.

        lon:
        lat:

        solar_charging: The amount of power entering the system via solar panels.
        power_load: The amount of power being used by the system.
        battery_charge: The amount of energy stored on the batteries.

        seaweed_mass: The amount of air being pumped by the altitude control
            system.

        last_command: The previous command executed by the balloon.
      """
    date_time: dt.datetime
    time_elapsed: dt.timedelta = dt.timedelta()

    lon: units.Distance = units.Distance(m=0)
    lat: units.Distance = units.Distance(m=0)

    solar_charging: units.Power = units.Power(watts=0)
    power_load: units.Power = units.Power(watts=0)
    battery_charge: units.Energy = units.Energy(watt_hours=0)

    seaweed_mass: units.Mass = units.Mass(kg=0)

class Platform:
    """A simulation of a seaweed platform.

    This class holds the system state vector and equations of motion
    (simulate_step) for simulating a seaweed platform.
    """

    def __init__(self, state: PlatformState):
        self.state = state

    def simulate_step(
        self,
        current_field: current_field,
        action: float,
        time_delta: dt.timedelta,
        stride: dt.timedelta = dt.timedelta(seconds=10),
    ) -> PlatformState:
        """Steps forward the simulation.

        This moves the balloon's state forward according to the dynamics of motion
        for a stratospheric balloon.

        Args:
          wind_vector: A vector corresponding to the wind to apply to the balloon.
          atmosphere: The atmospheric conditions the balloon is flying in.
          action: An AltitudeControlCommand for the system to take during this
            simulation step, i.e., up/down/stay.
          time_delta: How much time is elapsing during this step. Must be a multiple
            of stride.
          stride: The step size for the simulation of the balloon physics.
        """
        # check how much action is possible at battery charge
        effective_action = action

        # Position
        self.state.lon = self.state.lon
        self.state.lat = self.state.lat

        # Battery
        self.state.battery_charge = self.state.battery_charge

        # Seaweed Growth
        self.state.seaweed_mass = self.state.seaweed_mass

        return self.state



