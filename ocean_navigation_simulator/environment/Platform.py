"""Simulator for a seaweed platform.

Typical usage:

  current_field = ...
  platform = Platform()
  for _ in range(horizon / stride):
    command = ...
    platform.simulate_step(current_field, command, stride)
    print(platform.x, platform.y)
"""
import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import casadi as ca
import numpy as np

from ocean_navigation_simulator.data_sources import OceanCurrentSource
from ocean_navigation_simulator.data_sources.OceanCurrentSource.OceanCurrentVector import (
    OceanCurrentVector,
)
from ocean_navigation_simulator.data_sources.SeaweedGrowth.SeaweedGrowthSource import (
    SeaweedGrowthSource,
)
from ocean_navigation_simulator.data_sources.SolarIrradiance.SolarIrradianceSource import (
    SolarIrradianceSource,
)
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    PlatformStateSet,
)
from ocean_navigation_simulator.utils import units


@dataclass
class PlatformAction:
    """
    magntiude -> float, % of max
    direction -> float, radians
    """

    magnitude: float
    direction: float

    def __add__(self, other: "PlatformAction") -> "PlatformAction":
        """
        Computes the sum of two polar vectors without transforming back and forth
        between polar & cartesian coordinates
        Args:
            other: Another PlatformAction object to add

        Returns:
            PlatformAction: sum of two platform action objects
        """
        #
        # https://math.stackexchange.com/a/1365938
        added_mag = np.sqrt(
            self.magnitude**2
            + other.magnitude**2
            + 2 * self.magnitude * other.magnitude * np.cos(other.direction - self.direction)
        )
        added_angle = self.direction + np.arctan2(
            other.magnitude * np.sin(other.direction - self.direction),
            self.magnitude + other.magnitude * np.cos(other.direction - self.direction),
        )
        return PlatformAction(magnitude=added_mag, direction=added_angle)

    def scaling(self, constant: float) -> "PlatformAction":
        """To multiply the PlatformAction object by a constant, which in polar coordinates,
        results in multiplying only the magnitude of the vector

        Args:
            constant: The constant to multiply the vector PlatformAction

        Returns:
            PlatformAction: scaled by the constant
        """
        return PlatformAction(magnitude=self.magnitude * constant, direction=self.direction)

    def __array__(self):
        return np.array([self.magnitude, self.direction])

    def __len__(self):
        return self.__array__().shape[0]

    def __getitem__(self, item):
        return self.__array__()[item]

    def to_platform_action_set(self):
        return PlatformActionSet(action_set=[self])

    def to_xy_propulsion(self) -> np.ndarray:
        """Helper function to output a velocity vector in m/s (xy propulsion)
        Returns:
            np.ndarray:  [x_propulsion: propulsion in x direction in m/s
                          y_propulsion: propulsion in y direction in m/s]
        """
        x_propulsion = np.cos(self.direction) * self.magnitude
        y_propulsion = np.sin(self.direction) * self.magnitude
        return np.array([x_propulsion, y_propulsion])

    @staticmethod
    def from_xy_propulsion(x_propulsion: float, y_propulsion: float):
        """Helper function to initialize a PlatformAction based on xy actuation.
        Args:
            x_propulsion: propulsion in x direction in m/s
            y_propulsion: propulsion in y direction in m/s
        Returns:
            PlatformAction object
        """
        thrust = np.sqrt(
            x_propulsion**2 + y_propulsion**2
        )  # Calculating thrust from distance formula on input u
        heading = np.arctan2(y_propulsion, x_propulsion)  # Finds heading angle from input u
        return PlatformAction(magnitude=thrust, direction=heading)


@dataclass
class PlatformActionSet:
    """
    A class containing a list of PlatformAction to facilitate
    multi-agent handling for the simulation
    """

    action_set: List[PlatformAction]

    def __array__(self) -> np.ndarray:
        """rows are platforms and columns: [mag, dir]

        Returns:
            np.ndarray
        """
        return np.array(self.action_set)  #

    # TODO implement from_xy_propulsion for x and y given as arrays in the multi-agent setting


class Platform:
    """A simulation of a seaweed platform.

    This class holds the system state vector and equations of motion
    (simulate_step) for simulating a seaweed platform.
    """

    ocean_source: OceanCurrentSource = None
    solar_source: SolarIrradianceSource = None
    seaweed_source: SeaweedGrowthSource = None
    state_set: PlatformStateSet = None

    def __init__(
        self,
        platform_dict: Dict,
        ocean_source: OceanCurrentSource,
        use_geographic_coordinate_system: bool,
        solar_source: Optional[SolarIrradianceSource] = None,
        seaweed_source: Optional[SeaweedGrowthSource] = None,
    ):

        # initialize platform logger
        self.logger = logging.getLogger("arena.platform")

        # Set the major member variables
        self.platform_dict = platform_dict
        self.dt_in_s = platform_dict["dt_in_s"]
        self.ocean_source = ocean_source
        self.solar_source = solar_source
        self.seaweed_source = seaweed_source
        self.use_geographic_coordinate_system = use_geographic_coordinate_system

        self.model_battery = self.solar_source is not None
        self.model_seaweed = self.solar_source is not None and self.seaweed_source is not None

        # Set parameters for the Platform dynamics
        self.u_max = units.Velocity(mps=platform_dict["u_max_in_mps"])
        # Only when energy level is modelled
        if self.model_battery:
            self.battery_capacity = units.Energy(watt_hours=platform_dict["battery_cap_in_wh"])
            self.drag_factor = platform_dict["drag_factor"] / platform_dict["motor_efficiency"]
            self.solar_charge_factor = (
                platform_dict["solar_panel_size"] * platform_dict["solar_efficiency"]
            )
        else:
            self.battery_capacity = units.Energy(watt_hours=platform_dict["battery_cap_in_wh"])

        self.state_set, self.F_x_next = [None] * 2
        self.nb_platforms = None

    def set_state(self, states: PlatformStateSet):
        """Helper function to set the state directly.
        Args:
           platform_state_set: has to be a set of platforms otherwise nb_platforms is not well initialized
        """
        if self.nb_platforms is None:
            raise Exception(
                "The simulator dynamics need to be initialized first before calling this function"
            )
        u_curr = self.ocean_source.u_curr_func(
            np.stack((states.get_timestamp_arr(), states.lat.deg, states.lon.deg), axis=0)
        )
        v_curr = self.ocean_source.v_curr_func(
            np.stack((states.get_timestamp_arr(), states.lat.deg, states.lon.deg), axis=0)
        )

        states.replace_velocities(
            u_mps=np.array(u_curr).squeeze(), v_mps=np.array(v_curr).squeeze()
        )
        self.state_set = states
        return states

    def simulate_step(self, action: PlatformActionSet) -> PlatformStateSet:
        """Steps forward the simulation.
        This moves the platforms's state forward according to the dynamics of motion
        Args:
            action: PlatformActionSet object
        Return:
            state:  the next state as PlatformStateSet object.
        """

        # check if the cached dynamics need to be updated
        self.update_dynamics(self.state_set)
        state_numpy = np.array(
            self.F_x_next(
                self.state_set.to_casadi_input_dynamics_array(), np.array(action), self.dt_in_s
            )
        ).astype("float64")
        self.state_set = PlatformStateSet.from_numpy(state_numpy)

        return self.state_set

    def initialize_dynamics(self, states: PlatformStateSet):
        """Run at arena.reset() to load data and cache in casadi functions to run fast during simultion.
        Args:
            state: PlatformStateSet for which to load the data for caching in space and time
        """
        start = time.time()
        self.nb_platforms = len(states)
        if not type(states) is PlatformStateSet:
            raise TypeError("The simulator works with PlatformStateSet objects")
        if self.ocean_source is not None:
            self.ocean_source.update_casadi_dynamics(states)
        if self.solar_source is not None:
            self.solar_source.update_casadi_dynamics(states)
        if self.seaweed_source is not None:
            self.seaweed_source.set_casadi_function()
        self.F_x_next = self.get_casadi_dynamics()

        self.logger.info(f"Platform: Update Casadi + Dynamics ({time.time() - start:.1f}s)")

    def update_dynamics(self, state: PlatformStateSet):
        """Run in the step loop of arena to check if dynamics need to be updated.
        Args:
            state: PlatformState for which to check if cached dynamics need to be updated.
        """
        start = time.time()
        # Check and directly reload if data for the casadi function for ocean or solar simulation needs to be updated.
        ocean_change = (
            self.ocean_source is not None
            and self.ocean_source.check_for_casadi_dynamics_update(state)
        )
        solar_change = (
            self.solar_source is not None
            and self.solar_source.check_for_casadi_dynamics_update(state)
        )
        # propagate the newly cached functions in the source objects to F_x_next and seaweed source.
        if ocean_change or solar_change:
            if solar_change:
                self.seaweed_source.set_casadi_function()
            self.F_x_next = self.get_casadi_dynamics()
            self.logger.info(f"Platform: Update Casadi + Dynamics ({time.time() - start:.1f}s)")

    def get_casadi_dynamics(self):
        """Function to construct the F_x_next symbolic casadi function to be used in simulation."""
        # TODO: split up in three functions with varying level of complexities: 1) lat, lon, 2) adding battery, 3) adding seaweed mass
        # Note: lat, lon depend on battery if used
        ##### Symbolic Variables #####
        start = time.time()
        sym_lon_degree = ca.MX.sym("lon", self.nb_platforms, 1)  # in deg or m
        sym_lat_degree = ca.MX.sym("lat", self.nb_platforms, 1)  # in deg or m
        sym_time = ca.MX.sym("time", self.nb_platforms, 1)  # in posix
        sym_battery = ca.MX.sym("battery", self.nb_platforms, 1)  # in Joule
        sym_seaweed_mass = ca.MX.sym("seaweed_mass", self.nb_platforms, 1)  # in Kg
        sym_dt = ca.MX.sym("dt")  # in s
        sym_u_thrust = ca.MX.sym("u_thrust", self.nb_platforms, 1)  # in % of u_max
        sym_u_angle = ca.MX.sym("u_angle", self.nb_platforms, 1)  # in radians

        # Model Battery: Intermediate Variables for capping thrust so that battery never goes below 0
        if self.model_battery:
            sym_solar_charging = (
                self.solar_charge_factor
                * self.solar_source.solar_rad_casadi(
                    ca.horzcat(sym_time, sym_lat_degree, sym_lon_degree).T
                ).T
            )
            energy_available_in_J = sym_battery + sym_solar_charging * sym_dt
            energy_required_for_u_max = self.drag_factor * self.u_max.mps**3
            sym_u_thrust_capped = ca.fmin(
                sym_u_thrust, (energy_available_in_J / energy_required_for_u_max) ** (1.0 / 3)
            )
            sym_power_consumption = self.drag_factor * (self.u_max.mps * sym_u_thrust_capped) ** 3
        else:
            sym_solar_charging = 0
            sym_u_thrust_capped = sym_u_thrust
            sym_power_consumption = 0

        # Equations for m/s in latitude and longitude direction
        # For interpolation: need to pass a matrix (time, lat,lon) x nb_platforms (i.e. platforms as columns and not as rows)
        u_curr = self.ocean_source.u_curr_func(
            ca.horzcat(sym_time, sym_lat_degree, sym_lon_degree).T
        ).T  # retranspose it back to a vector where platforms are rows
        v_curr = self.ocean_source.v_curr_func(
            ca.horzcat(sym_time, sym_lat_degree, sym_lon_degree).T
        ).T
        sym_lon_delta_meters_per_s = (
            ca.cos(sym_u_angle) * sym_u_thrust_capped * self.u_max.mps + u_curr
        )
        sym_lat_delta_meters_per_s = (
            ca.sin(sym_u_angle) * sym_u_thrust_capped * self.u_max.mps + v_curr
        )

        # Transform the delta_meters from propulsion to the global coordinate system used.
        if self.use_geographic_coordinate_system:
            # Equations for delta in latitude and longitude direction in degree
            sym_lon_delta_deg_per_s = (
                180
                * sym_lon_delta_meters_per_s
                / math.pi
                / 6371000
                / ca.cos(math.pi * sym_lat_degree / 180)
            )
            sym_lat_delta_deg_per_s = 180 * sym_lat_delta_meters_per_s / math.pi / 6371000
        else:  # Global coordinate system in meters
            sym_lon_delta_deg_per_s = sym_lon_delta_meters_per_s
            sym_lat_delta_deg_per_s = sym_lat_delta_meters_per_s

        # Model Seaweed Growth
        if self.model_seaweed:
            sym_growth_factor = self.seaweed_source.F_NGR_per_second(
                ca.horzcat(sym_time, sym_lat_degree, sym_lon_degree).T
            ).T
        else:
            sym_growth_factor = 0

        # Equations for next states using the intermediate variables from above
        sym_lon_next = sym_lon_degree + sym_dt * sym_lon_delta_deg_per_s
        sym_lat_next = sym_lat_degree + sym_dt * sym_lat_delta_deg_per_s
        sym_battery_next = ca.fmin(
            self.battery_capacity.joule,
            ca.fmax(0, sym_battery + sym_dt * (sym_solar_charging - sym_power_consumption)),
        )
        sym_time_next = sym_time + sym_dt
        sym_seaweed_mass_next = sym_seaweed_mass + sym_dt * sym_growth_factor * sym_seaweed_mass

        F_next = ca.Function(
            "F_x_next",
            [
                ca.horzcat(sym_lon_degree, sym_lat_degree, sym_time, sym_battery, sym_seaweed_mass),
                ca.horzcat(sym_u_thrust, sym_u_angle),
                sym_dt,
            ],
            [
                ca.horzcat(
                    sym_lon_next,
                    sym_lat_next,
                    sym_time_next,
                    sym_battery_next,
                    sym_seaweed_mass_next,
                    sym_lon_delta_meters_per_s,
                    sym_lat_delta_meters_per_s,
                )
            ],
        )

        self.logger.info(f"Platform: Set Dynamics F_x_next Function ({time.time() - start:.1f}s)")
        return F_next
