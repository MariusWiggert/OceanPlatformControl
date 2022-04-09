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
class PlatformConfig(object):
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
    battery_cap: units.Energy()
    u_max: units.Velocity
    motor_efficiency: float


#platform_config_dict = {'battery_cap': 400.0, 'u_max': 0.1, 'motor_efficiency': 1.0,
#                        'solar_panel_size': 0.5, 'solar_efficiency': 0.2, 'drag_factor': 675}

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

    def __init__(self, state: PlatformState, source: OceanCurrentSource):
        self.state = state

        #data_store = utils.simulation_utils.copernicusmarine_datastore('cmems_mod_glo_phy_anfc_merged-uv_PT1H-i', 'mmariuswiggert', 'tamku3-qetroR-guwneq')
        #DS_currents = xr.open_dataset(data_store)[['uo', 'vo']].isel(depth=0)
        #self.old_source = {'data_source_type': 'cop_opendap', 'content': DS_currents, 'grid_dict': Problem.derive_grid_dict_from_xarray(DS_currents)}

        self.source = source

        self.sim_settings = {
            'deg_around_x_t': 2,
            'ti': 2,
            'hours_to_sim_timescale': 3600,
            't_horizon_sim': 10,
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

        self.F_x_next = None
        self.prev_Dt = None

    def simulate_step(
        self,
        action: float,
        time_delta: dt.timedelta,
        stride: dt.timedelta,
    ) -> PlatformState:
        """Steps forward the simulation.

        This moves the balloon's state forward according to the dynamics of motion
        for a stratospheric balloon.

        Args:
          action: An AltitudeControlCommand for the system to take during this
            simulation step, i.e., up/down/stay.
          time_delta: How much time is elapsing during this step. Must be a multiple
            of stride.
          stride: The step size for the simulation of the balloon physics.
        """
        self.sim_settings['dt'] = time_delta.seconds
        self.cur_state = np.array([self.state.lon.deg, self.state.lat.deg, self.state.battery_charge.watt_hours, self.state.date_time.timestamp()])

        start = time.time()
        self.check_dynamics_update(time_delta)
        #print(f'Update Dynamics: {time.time()-start}s')
        start = time.time()
        self.run_step()
        #print(f'Run Step: {time.time()-start}s')

        #print(self.cur_state)
        self.state.lon = units.Distance(deg=self.cur_state[0])
        self.state.lat = units.Distance(deg=self.cur_state[1])
        self.state.battery_charge = units.Energy(watt_hours=self.cur_state[2])
        self.state.date_time = dt.datetime.fromtimestamp(self.cur_state[3])

        return self.state

    def check_dynamics_update(self, time_delta):
        """ Helper function for main loop to check if we need to load new current data into the dynamics."""
        if time_delta != self.prev_Dt:
            start = time.time()
            self.update_dynamics(self.cur_state.flatten())
            print(f'Update Dynamics: {time.time()-start}s')
        else:
            # check based on space and time of the currently loaded current data if new data needs to be loaded
            if not (self.lon_low < self.cur_state[0] < self.lon_high) or not (self.lat_low < self.cur_state[1] < self.lat_high) or not (self.t_low < self.cur_state[3] < self.t_high):
                print()
                print("Updating simulator dynamics with new current data.")
                if not (self.lon_low < self.cur_state[0] < self.lon_high):
                    print('Lon:', self.lon_low, self.cur_state[0], self.lon_high)
                if not not (self.lat_low < self.cur_state[1] < self.lat_high):
                    print('Lat:', self.lat_low, self.cur_state[1], self.lat_high)
                if not not (self.t_low < self.cur_state[3] < self.t_high):
                    print('Time:', self.t_low, self.cur_state[3], self.t_high)
                print()
                start = time.time()
                self.update_dynamics(self.cur_state.flatten())
                print(f'Update Dynamics Inner: {time.time()-start}s')

        self.prev_Dt = time_delta
            
    def update_dynamics(self, x_t):
        """Update symbolic dynamics function for the simulation by sub-setting the relevant set of current data.
        Specifically, the self.F_x_next symbolic function and the self.grid_dict
        Args:
            x_t: current location of agent
        Output:
            F_x_next          cassadi function with (x,u)->(x_next)
        """
        lon_interval = [x_t[0] - self.sim_settings["deg_around_x_t"], x_t[0] + self.sim_settings["deg_around_x_t"]]
        lat_interval = [x_t[1] - self.sim_settings["deg_around_x_t"], x_t[1] + self.sim_settings["deg_around_x_t"]]
        t_interval = [dt.datetime.fromtimestamp(x_t[3] - 3600, tz=dt.timezone.utc), dt.datetime.fromtimestamp(x_t[3] + 3600 * 24 * 10, tz=dt.timezone.utc)]

        ##### Get Data #####
        start = time.time()
        self.data = self.source.get_data_over_area(x_interval=lon_interval, y_interval=lat_interval, t_interval=t_interval)
        self.grid = [
            (self.data.coords['time'].values - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'),
            self.data.coords['lat'].values,
            self.data.coords['lon'].values
        ]
        self.lon_low, self.lon_high = self.grid[2].min(), self.grid[2].max()
        self.lat_low, self.lat_high = self.grid[1].min(), self.grid[1].max()
        self.t_low, self.t_high = self.grid[0].min(), self.grid[0].max()
        print(f'Get Data I: {time.time()-start}s')


        ##### Interpolation #####
        start = time.time()
        u_curr_func = ca.interpolant('u_curr', 'linear', self.grid, self.data['water_u'].values.ravel(order='F'))
        v_curr_func = ca.interpolant('v_curr', 'linear', self.grid, self.data['water_v'].values.ravel(order='F'))
        print(f'Set Up Interpolation: {time.time()-start}s')

        ##### Equations #####
        start = time.time()
        sym_lon = ca.MX.sym('x1')  # lon
        sym_lat = ca.MX.sym('x2')  # lat
        sym_battery = ca.MX.sym('x3')  # battery
        sym_time = ca.MX.sym('t')  # time
        u_sim_1 = ca.MX.sym('u_1')  # thrust magnitude in [0,1]
        u_sim_2 = ca.MX.sym('u_2')  # header in radians

        x_sym = ca.vertcat(sym_lon, sym_lat, sym_battery, sym_time)
        u_sym = ca.vertcat(u_sim_1, u_sim_2)

        lon_dot = (ca.cos(u_sim_2) * u_sim_1 * self.u_max + u_curr_func(ca.vertcat(sym_time, sym_lat, sym_lon))) / self.sim_settings['conv_m_to_deg']
        lat_dot = (ca.sin(u_sim_2) * u_sim_1 * self.u_max + v_curr_func(ca.vertcat(sym_time, sym_lat, sym_lon))) / self.sim_settings['conv_m_to_deg']
        battery_dot = self.charge_factor * solar_rad(sym_time, sym_lat, sym_lon) - self.energy_coeff * (self.u_max * u_sim_1) ** 3
        time_dot = 1

        self.F_x_next = ca.Function(
            'F_x_next',
            [x_sym, u_sym],
            [x_sym + self.sim_settings['dt'] * ca.vertcat(lon_dot, lat_dot, battery_dot, time_dot)],
        )
        print(f'Equations: {time.time()-start}s')

    def run_step(self):
        """Run the simulator for one dt step"""

        #u = self.thrust_check(self.steering_controller.get_next_action(self.cur_state, self.trajectory))
        u = np.array([1,3.14/2])
        self.cur_state = np.array(self.F_x_next(self.cur_state, u)).astype('float64').flatten()


    def thrust_check(self, u_planner):
        """If the thrust would use more energy than available adjust accordingly."""

        if np.isnan(u_planner).sum() > 0 :
            raise ValueError("Planner action contains Nan, specifically u_planner=", u_planner)

        charge = self.charge_factor * solar_rad(self.cur_state[3], self.cur_state[1], self.cur_state[0])

        delta_charge = charge -  self.energy_coeff * (self.u_max * u_planner[0]) ** 3

        next_charge = self.cur_state[2] + delta_charge * self.sim_settings['dt']

        # if smaller than 0.: change the thrust accordingly
        if next_charge < 0.:
            energy_available = self.cur_state[2]
            u_planner[0] = ((charge - energy_available / dt) / (self.energy_coeff * self.u_max ** 3)) ** (1. / 3)
            return u_planner
        else:
            return u_planner


    @staticmethod
    def battery_check(cur_state):
        """Prevents battery level to go above 1."""
        if cur_state[2] > 1.:
            cur_state[2] = 1.
        return cur_state

