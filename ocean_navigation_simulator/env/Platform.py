"""Simulator for a seaweed platform.

Typical usage:

  current_field = ...
  platform = Platform()
  for _ in range(horizon / stride):
    command = ...
    platform.simulate_step(current_field, command, stride)
    print(platform.x, platform.y)
"""
from dataclasses import dataclass
import casadi as ca
import numpy as np
import time
import math
from typing import Dict, Optional

from ocean_navigation_simulator.env.data_sources.OceanCurrentSource.OceanCurrentSource import OceanCurrentSource
from ocean_navigation_simulator.env.data_sources.SolarIrradiance.SolarIrradianceSource import SolarIrradianceSource
from ocean_navigation_simulator.env.data_sources.SeaweedGrowth.SeaweedGrowthSource import SeaweedGrowthSource
from ocean_navigation_simulator.env.utils import units
from ocean_navigation_simulator.env.PlatformState import PlatformState


@dataclass
class PlatformAction():
    """
    magntiude -> float, % of max
    direction -> float, radians
    """
    magnitude: float
    direction: float

    def __array__(self):
        return np.array([self.magnitude, self.direction])

    def __len__(self):
        return self.__array__().shape[0]

    def __getitem__(self, item):
        return self.__array__()[item]


    @staticmethod
    def from_xy_propulsion(x_propulsion: float, y_propulsion: float):
        """Helper function to initialize a PlatformAction based on xy actuation.
        Args:
            x_propulsion: propulsion in x direction in m/s
            y_propulsion: propulsion in y direction in m/s
        Returns:
            PlatformAction object
        """
        thrust = np.sqrt(x_propulsion ** 2 + y_propulsion ** 2)  # Calculating thrust from distance formula on input u
        heading = np.arctan2(y_propulsion, x_propulsion)  # Finds heading angle from input u
        return PlatformAction(magnitude=thrust, direction=heading)


class Platform:
    """A simulation of a seaweed platform.

    This class holds the system state vector and equations of motion
    (simulate_step) for simulating a seaweed platform.
    """

    def __init__(self,
                 platform_dict: Dict,
                 ocean_source: OceanCurrentSource,
                 use_geographic_coordinate_system: bool,
                 solar_source: Optional[SolarIrradianceSource] = None,
                 seaweed_source: Optional[SeaweedGrowthSource] = None):

        # Set the major member variables
        self.dt_in_s = platform_dict['dt_in_s']
        self.ocean_source = ocean_source
        self.solar_source = solar_source
        self.seaweed_source = seaweed_source
        self.use_geographic_coordinate_system = use_geographic_coordinate_system

        self.model_battery = self.solar_source is not None
        self.model_seaweed = self.solar_source is not None and self.seaweed_source is not None

        # Set parameters for the Platform dynamics
        self.u_max = units.Velocity(mps=platform_dict['u_max_in_mps'])
        # Only when energy level is modelled
        if self.model_battery:
            self.battery_capacity = units.Energy(watt_hours=platform_dict['battery_cap_in_wh'])
            self.drag_factor = platform_dict['drag_factor'] / platform_dict['motor_efficiency']
            self.solar_charge_factor = platform_dict['solar_panel_size'] * platform_dict['solar_efficiency']
        else:
            self.battery_capacity = units.Energy(watt_hours=platform_dict['battery_cap_in_wh'])

        self.state, self.F_x_next = [None]*2

    def set_state(self, state: PlatformState):
        """Helper function to set the state."""
        self.state = state

    def simulate_step(self, action: PlatformAction) -> PlatformState:
        """Steps forward the simulation.

        This moves the platforms's state forward according to the dynamics of motion

        Args:
          action:
          time_delta: How much time is elapsing during this step.
          intermediate_steps:
        """
        # check if the cached dynamics need to be updated
        self.update_dynamics(self.state)

        state_numpy = np.array(self.F_x_next(np.array(self.state), np.array(action), self.dt_in_s)).astype('float64').flatten()
        self.state = PlatformState.from_numpy(state_numpy)

        return self.state

    def initialize_dynamics(self, state: PlatformState):
        """Run at arena.reset()"""
        start = time.time()
        if self.ocean_source is not None:
            self.ocean_source.update_casadi_dynamics(state)
        if self.solar_source is not None:
            self.solar_source.update_casadi_dynamics(state)
        if self.seaweed_source is not None:
            self.seaweed_source.set_casadi_function()
        self.F_x_next = self.get_casadi_dynamics()
        #print(f'Initialize Casadi + Dynamics: {time.time() - start:.2f}s')

    def update_dynamics(self, state: PlatformState):
        """Run in the step loop of arena."""
        start = time.time()
        ocean_change = (self.ocean_source is not None and self.ocean_source.check_for_casadi_dynamics_update(state))
        solar_change = (self.solar_source is not None and self.solar_source.check_for_casadi_dynamics_update(state))
        if ocean_change or solar_change:
            if solar_change:
                self.seaweed_source.set_casadi_function()
            self.F_x_next = self.get_casadi_dynamics()
            print(f'Update Casadi + Dynamics: {time.time() - start:.2f}s')

    def get_casadi_dynamics(self):
        # TODO: split up in three functions with varying level of complexities: 1) lat, lon, 2) adding battery, 3) adding seaweed mass
        # Could also be done with subclassing of Platform.
        ##### Equations #####
        start = time.time()
        sym_lon_degree      = ca.MX.sym('lon')          # in deg or m
        sym_lat_degree      = ca.MX.sym('lat')          # in deg or m
        sym_time            = ca.MX.sym('time')         # in posix
        sym_battery         = ca.MX.sym('battery')      # in Joule
        sym_seaweed_mass    = ca.MX.sym('battery')      # in Kg
        sym_dt              = ca.MX.sym('dt')           # in s
        sym_u_thrust        = ca.MX.sym('u_thrust')     # in % of u_max
        sym_u_angle         = ca.MX.sym('u_angle')      # in radians

        # Model Battery: Intermediate Variables for capping thrust so that battery never goes below 0
        if self.model_battery:
            sym_solar_charging = self.solar_charge_factor * self.solar_source.solar_rad_casadi(
                ca.vertcat(sym_time, sym_lat_degree, sym_lon_degree))
            energy_available_in_J = sym_battery + sym_solar_charging * sym_dt
            energy_required_for_u_max = self.drag_factor * self.u_max.mps ** 3
            sym_u_thrust_capped = ca.fmin( sym_u_thrust,(energy_available_in_J/energy_required_for_u_max) ** (1./3))
            sym_power_consumption = self.drag_factor * (self.u_max.mps * sym_u_thrust_capped) ** 3
        else:
            sym_solar_charging = 0
            sym_u_thrust_capped = sym_u_thrust
            sym_power_consumption = 0

        # Equations for m/s in latitude and longitude direction
        u_curr = self.ocean_source.u_curr_func(ca.vertcat(sym_time, sym_lat_degree, sym_lon_degree))
        v_curr = self.ocean_source.v_curr_func(ca.vertcat(sym_time, sym_lat_degree, sym_lon_degree))
        sym_lon_delta_meters = ca.cos(sym_u_angle) * sym_u_thrust_capped * self.u_max.mps + u_curr
        sym_lat_delta_meters = ca.sin(sym_u_angle) * sym_u_thrust_capped * self.u_max.mps + v_curr

        # Transform the delta_meters from propulsion to the global coordinate system used.
        if self.use_geographic_coordinate_system:
            # Equations for delta in latitude and longitude direction in degree
            sym_lon_delta = 180 * sym_lon_delta_meters / math.pi / 6371000 / ca.cos(math.pi * sym_lat_degree / 180)
            sym_lat_delta = 180 * sym_lat_delta_meters / math.pi / 6371000
        else: # Global coordinate system in meters
            sym_lon_delta = sym_lon_delta_meters
            sym_lat_delta = sym_lat_delta_meters

        # Model Seaweed Growth
        if self.model_seaweed:
            sym_growth_factor = self.seaweed_source.F_NGR_per_second(ca.vertcat(sym_time, sym_lat_degree, sym_lon_degree))
        else:
            sym_growth_factor = 0

        # Equations for next states using the intermediate variables from above
        sym_lon_next = sym_lon_degree + sym_dt * sym_lon_delta
        sym_lat_next = sym_lat_degree + sym_dt * sym_lat_delta
        sym_battery_next = ca.fmin(self.battery_capacity.joule,
                                   ca.fmax(0, sym_battery + sym_dt * (sym_solar_charging - sym_power_consumption)))
        sym_time_next = sym_time + sym_dt
        sym_seaweed_mass_next = sym_seaweed_mass + sym_dt * sym_growth_factor * sym_seaweed_mass

        F_next = ca.Function(
            'F_x_next',
            [ca.vertcat(sym_lon_degree, sym_lat_degree, sym_time, sym_battery, sym_seaweed_mass), ca.vertcat(sym_u_thrust, sym_u_angle), sym_dt],
            [ca.vertcat(sym_lon_next, sym_lat_next, sym_time_next, sym_battery_next, sym_seaweed_mass_next)],
        )
        #print(f'Set Platform Equations: {time.time() - start:.2f}s')
        return F_next
