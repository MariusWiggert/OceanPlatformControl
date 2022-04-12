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
import datetime as dt
import casadi as ca
import numpy as np
import time
import math


from ocean_navigation_simulator.env.data_sources.OceanCurrentSource.OceanCurrentSource import OceanCurrentSource
from ocean_navigation_simulator.env.data_sources.SolarIrradiance.SolarIrradianceSource import SolarIrradianceSource
from ocean_navigation_simulator.env.utils import units

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

    lon: units.Distance
    lat: units.Distance

    battery_charge: units.Energy = units.Energy(watt_hours=0)

    seaweed_mass: units.Mass = units.Mass(kg=0)


@dataclasses.dataclass
class PlatformSpecs(object):
    """A dataclass containing constants relevant to the platform state.

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
    battery_cap: units.Energy(watt_hours=0)
    u_max: units.Velocity(meters_per_second=0)
    motor_efficiency: float


#platform_config_dict = {'battery_cap': 400.0, 'u_max': 0.1, 'motor_efficiency': 1.0,
#                        'solar_panel_size': 0.5, 'solar_efficiency': 0.2, 'drag_factor': 675}

class Platform:
    """A simulation of a seaweed platform.

    This class holds the system state vector and equations of motion
    (simulate_step) for simulating a seaweed platform.
    """

    def __init__(self, state: PlatformState, specs: PlatformSpecs, ocean_source: OceanCurrentSource, solar_source: SolarIrradianceSource):
        self.state = state
        self.specs = specs
        self.ocean_source = ocean_source
        self.solar_source = solar_source

        platform_specs = {
            'battery_cap_in_wh': 400.0,
            'u_max_in_mps': 0.1,
            'motor_efficiency': 1.0,
            'solar_panel_size': 0.5,
            'solar_efficiency': 0.2,
            'drag_factor': 675
        }
        cap_in_joule = platform_specs['battery_cap_in_wh'] * 3600
        self.energy_coeff = (platform_specs['drag_factor'] * (1 / platform_specs['motor_efficiency'])) / cap_in_joule
        self.charge_factor =  (platform_specs['solar_panel_size'] * platform_specs['solar_efficiency']) / cap_in_joule
        self.u_max = platform_specs['u_max_in_mps']

    def simulate_step(
        self,
        action: tuple[float, float],
        time_delta: dt.timedelta,
        intermediate_steps: int = 1,
    ) -> PlatformState:
        """Steps forward the simulation.

        This moves the platforms's state forward according to the dynamics of motion

        Args:
          action:
          time_delta: How much time is elapsing during this step.
          intermediate_steps:
        """
        state_numpy = np.array([self.state.lon.deg, self.state.lat.deg, self.state.date_time.timestamp(), self.state.battery_charge.watt_hours])

        for i in range(intermediate_steps):
            self.update_dynamics(state_numpy)
            state_numpy = np.array(self.F_x_next(state_numpy, np.array(action), time_delta.seconds / intermediate_steps)).astype('float64').flatten()

        self.state.lon              = units.Distance(deg=state_numpy[0])
        self.state.lat              = units.Distance(deg=state_numpy[1])
        self.state.date_time        = dt.datetime.fromtimestamp(state_numpy[2])
        self.state.battery_charge   = units.Energy(watt_hours=state_numpy[3])

        return self.state

    def update_dynamics(self, state_numpy) -> ca.Function:
        start = time.time()

        casadi_state = [state_numpy[0], state_numpy[1], state_numpy[3], state_numpy[2]]
        if not hasattr(self, 'F_x_next'):
            self.solar_source.update_casadi_dynamics(casadi_state)
            self.ocean_source.update_casadi_dynamics(casadi_state)
            self.F_x_next = self.get_dynamics(state_numpy)
            print(f'Initialize Casadi + Dynamics: {time.time()-start:.2f}s')
        elif self.solar_source.check_for_casadi_dynamics_update(casadi_state) or self.ocean_source.check_for_casadi_dynamics_update(casadi_state):
            self.F_x_next = self.get_dynamics(state_numpy)
            print(f'Update Casadi + Dynamics: {time.time()-start:.2f}s')

    def get_dynamics(self, state_numpy):
        ##### Equations #####
        start = time.time()
        sym_lon_degree      = ca.MX.sym('lon')
        sym_lat_degree      = ca.MX.sym('lat')
        sym_time            = ca.MX.sym('time')
        sym_battery         = ca.MX.sym('battery')
        sym_dt              = ca.MX.sym('dt')
        sym_u_thrust        = ca.MX.sym('u_thrust')
        sym_u_angle         = ca.MX.sym('u_angle')

        sym_solar_charging = self.charge_factor * self.solar_source.solar_rad_casadi(ca.vertcat(sym_time, sym_lat_degree, sym_lon_degree))
        sym_u_thrust_capped = ca.fmin(sym_u_thrust, ((sym_battery + sym_solar_charging / sym_dt) / (self.energy_coeff * self.specs.u_max.mps ** 3)) ** (1. / 3))
        sym_power_load = self.energy_coeff * (self.specs.u_max.mps * sym_u_thrust_capped) ** 3

        sym_lon_delta_meters = ca.cos(sym_u_angle) * sym_u_thrust_capped * self.specs.u_max.mps + self.ocean_source.u_curr_func(ca.vertcat(sym_time, sym_lat_degree, sym_lon_degree))
        sym_lat_delta_meters = ca.sin(sym_u_angle) * sym_u_thrust_capped * self.specs.u_max.mps + self.ocean_source.v_curr_func(ca.vertcat(sym_time, sym_lat_degree, sym_lon_degree))

        sym_lon_delta_degree = 180 * sym_lon_delta_meters / math.pi / 6371000 / ca.cos(math.pi * sym_lat_degree / 180)
        sym_lat_delta_degree = 180 * sym_lat_delta_meters / math.pi / 6371000

        sym_lon_next = sym_lon_degree + sym_dt * sym_lon_delta_degree
        sym_lat_next = sym_lat_degree + sym_dt * sym_lat_delta_degree
        sym_battery_next = ca.fmin(1, ca.fmax(0, sym_battery + sym_dt * (sym_solar_charging - sym_power_load)))
        sym_time_next = sym_time + sym_dt

        F_next = ca.Function(
            'F_x_next',
            [ca.vertcat(sym_lon_degree, sym_lat_degree, sym_time, sym_battery), ca.vertcat(sym_u_thrust, sym_u_angle), sym_dt],
            [ca.vertcat(sym_lon_next, sym_lat_next, sym_time_next, sym_battery_next)],
        )
        print(f'Set Platform Equations: {time.time()-start:.2f}s')
        return F_next