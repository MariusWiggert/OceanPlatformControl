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
import xarray as xr
import time

from ocean_navigation_simulator import Problem
from ocean_navigation_simulator.data_sources.OceanCurrentSource.OceanCurrentSource import HindcastOpendapSource, OceanCurrentSource
from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator import utils

from ocean_navigation_simulator.utils import plotting_utils, simulation_utils, solar_rad

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

    def __init__(self, state: PlatformState, specs: PlatformSpecs, source: OceanCurrentSource):
        self.state = state
        self.specs = specs
        self.source = source

        self.sim_settings = {
            'deg_around_x_t': 2,
            'time_around_x_t': 3600 * 24 * 10,
            'conv_m_to_deg': 111120.
        }
        platform_specs = {
            'battery_cap': 400.0,
            'u_max': 0.1,
            'motor_efficiency': 1.0,
            'solar_panel_size': 0.5,
            'solar_efficiency': 0.2,
            'drag_factor': 675
        }
        cap_in_joule = platform_specs['battery_cap'] * 3600
        self.energy_coeff = (platform_specs['drag_factor'] * (1 / platform_specs['motor_efficiency'])) / cap_in_joule
        self.charge_factor =  (platform_specs['solar_panel_size'] * platform_specs['solar_efficiency']) / cap_in_joule
        self.u_max = platform_specs['u_max']

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
        state_numpy = np.array([self.state.lon.deg, self.state.lat.deg, self.state.battery_charge.watt_hours, self.state.date_time.timestamp()])
        u_numpy = np.array(action)

        for i in range(intermediate_steps):
            self.update_dynamics(state_numpy)
            state_numpy = np.array(self.F_x_next(state_numpy, u_numpy, time_delta.seconds / intermediate_steps)).astype('float64').flatten()

        self.state.lon = units.Distance(deg=state_numpy[0])
        self.state.lat = units.Distance(deg=state_numpy[1])
        self.state.battery_charge = units.Energy(watt_hours=state_numpy[2])
        self.state.date_time = dt.datetime.fromtimestamp(state_numpy[3])

        return self.state

    def update_dynamics(self, state_numpy) -> ca.Function:
        if not hasattr(self, 'F_x_next'):
            start = time.time()
            self.F_x_next = self.get_dynamics(state_numpy)
            print(f'Initialize Dynamics: {time.time()-start}s')
        elif not (self.lon_low < state_numpy[0] < self.lon_high) or not (self.lat_low < state_numpy[1] < self.lat_high) or not (self.t_low < state_numpy[3] < self.t_high):
            if not (self.lon_low < state_numpy[0] < self.lon_high):
                print(f'Updating Dynamics (Lon: {self.lon_low}, {state_numpy[0]}, {self.lon_high})')
            if not not (self.lat_low < state_numpy[1] < self.lat_high):
                print(f'Updating Dynamics (Lat: {self.lat_low}, {state_numpy[1]}, {self.lat_high})')
            if not not (self.t_low < state_numpy[3] < self.t_high):
                print(f'Updating Dynamics (Time: {self.t_low}, {state_numpy[3]}, {self.t_high})')
            start = time.time()
            self.F_x_next = self.get_dynamics(state_numpy)
            print(f'Update Dynamics: {time.time()-start}s')

    def get_dynamics(self, state_numpy):
        ##### Intervals #####
        lon_interval = [state_numpy[0] - self.sim_settings["deg_around_x_t"], state_numpy[0] + self.sim_settings["deg_around_x_t"]]
        lat_interval = [state_numpy[1] - self.sim_settings["deg_around_x_t"], state_numpy[1] + self.sim_settings["deg_around_x_t"]]
        t_interval = [dt.datetime.fromtimestamp(state_numpy[3] - 3600, tz=dt.timezone.utc), dt.datetime.fromtimestamp(state_numpy[3] + self.sim_settings['time_around_x_t'], tz=dt.timezone.utc)]

        ##### Get Data #####
        start = time.time()
        self.data = self.source.get_currents_over_area(x_interval=lon_interval, y_interval=lat_interval, t_interval=t_interval)
        self.grid = [
            (self.data.coords['time'].values - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'),
            self.data.coords['lat'].values,
            self.data.coords['lon'].values
        ]
        self.lon_low, self.lon_high = self.grid[2].min(), self.grid[2].max()
        self.lat_low, self.lat_high = self.grid[1].min(), self.grid[1].max()
        self.t_low, self.t_high = self.grid[0].min(), self.grid[0].max()
        print()
        print(f'Get Data: {time.time()-start}s')

        ##### Interpolation #####
        start = time.time()
        u_curr_func = ca.interpolant('u_curr', 'linear', self.grid, self.data['water_u'].values.ravel(order='F'))
        v_curr_func = ca.interpolant('v_curr', 'linear', self.grid, self.data['water_v'].values.ravel(order='F'))
        print(f'Set Up Interpolation: {time.time()-start}s')

        ##### Equations #####
        start = time.time()
        sym_lon     = ca.MX.sym('x1')
        sym_lat     = ca.MX.sym('x2')
        sym_battery = ca.MX.sym('x3')
        sym_time    = ca.MX.sym('t')
        sym_dt      = ca.MX.sym('dt')
        sym_u_thrust= ca.MX.sym('u_1')
        sym_u_angle = ca.MX.sym('u_2')

        sym_charge = self.charge_factor * solar_rad(sym_time, sym_lat, sym_lon)
        sym_u_thrust_capped = ca.fmin(sym_u_thrust, ((sym_battery + sym_charge / sym_dt) / (self.energy_coeff * self.specs.u_max.mps ** 3)) ** (1. / 3))

        sym_lon_next = sym_lon + sym_dt * (ca.cos(sym_u_angle) * sym_u_thrust_capped * self.specs.u_max.mps + u_curr_func(ca.vertcat(sym_time, sym_lat, sym_lon))) / self.sim_settings['conv_m_to_deg']
        sym_lat_next = sym_lat + sym_dt * (ca.sin(sym_u_angle) * sym_u_thrust_capped * self.specs.u_max.mps + v_curr_func(ca.vertcat(sym_time, sym_lat, sym_lon))) / self.sim_settings['conv_m_to_deg']
        sym_battery_next = ca.fmin(1, ca.fmax(0, sym_battery + sym_dt * (sym_charge - self.energy_coeff * (self.specs.u_max.mps * sym_u_thrust_capped) ** 3)))
        sym_time_next = sym_time + sym_dt

        F_next = ca.Function(
            'F_x_next',
            [ca.vertcat(sym_lon, sym_lat, sym_battery, sym_time), ca.vertcat(sym_u_thrust, sym_u_angle), sym_dt],
            [ca.vertcat(sym_lon_next, sym_lat_next, sym_battery_next, sym_time_next)],
        )
        print(f'Set Equations: {time.time()-start}s')
        return F_next