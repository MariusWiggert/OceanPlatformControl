"""A Ocean Platform Arena.
A Ocean arena contains the logic for navigating a platform in the ocean.
"""

import dataclasses
from typing import Dict, Optional
import numpy as np
import xarray as xr
from ocean_navigation_simulator import Problem

from ocean_navigation_simulator.env.data_sources.OceanCurrentField import OceanCurrentField
from ocean_navigation_simulator.env.data_sources.SolarIrradianceField import SolarIrradianceField
from ocean_navigation_simulator.env.data_sources.SeaweedGrowthField import SeaweedGrowthField
from ocean_navigation_simulator.env.Platform import Platform, PlatformState, PlatformAction
from ocean_navigation_simulator.utils import plotting_utils, simulation_utils
import ocean_navigation_simulator.env.utils.units as units


@dataclasses.dataclass
class ArenaObservation:
    """
    Specifies an observation from the simulator.
    This differs from SimulatorState in that the observations are not
    ground truth state, and are instead noisy observations from the
    environment.
    """
    platform_state: PlatformState     # position, time, battery
    #current_at_platform: OceanCurrentVector  # real current at platfrom
    #forecasts: np.ndarray                # solar, current local current


class Arena:
    """A OceanPlatformArena in which an ocean platform moves through a current field."""

    # TODO: where do we do the reset? I guess for us reset mostly would mean new start and goal position?
    # TODO: not sure what that should be for us, decide where to put the feature constructor
    def __init__(self, sim_cache_dict: Dict, platform_dict: Dict, ocean_dict: Dict,
                 solar_dict: Optional[Dict] = None, seaweed_dict: Optional[Dict] = None,
                 geographic_coordinate_system: Optional[bool] = True):
        """OceanPlatformArena constructor.
    Args:
        sim_cache_dict:
        platform_dict:
        ocean_dict:
        solar_dict:
    Optional Args:
        geographic_coordinate_system: If True we use the Geographic coordinate system in lat, lon degree, if false the spatial system is in meters in x, y.
    """

        # Initialize the Data Fields from the respective dictionaries
        self.ocean_field = OceanCurrentField(sim_cache_dict=sim_cache_dict,
                                             hindcast_source_dict=ocean_dict['hindcast'],
                                             forecast_source_dict=ocean_dict['forecast'])
        if solar_dict is not None:
            self.solar_field = SolarIrradianceField(sim_cache_dict=sim_cache_dict,
                                                    hindcast_source_dict=solar_dict['hindcast'],
                                                    forecast_source_dict=solar_dict['forecast'])
        if seaweed_dict is not None:
            # For initializing the SeaweedGrowth Field we need to supply the respective SolarIrradianceSources
            seaweed_dict['hindcast']['solar_source'] = self.solar_field.hindcast_data_source
            if seaweed_dict['forecast'] is not None:
                seaweed_dict['forecast']['solar_source'] = self.solar_field.hindcast_data_source
            self.seaweed_field = SeaweedGrowthField(sim_cache_dict=sim_cache_dict,
                                                    hindcast_source_dict=seaweed_dict['hindcast'],
                                                    forecast_source_dict=seaweed_dict['forecast'])

        # Initialize the Platform Object from the dictionary
        self.platform = Platform(platform_dict=platform_dict,
                                 ocean_source=self.ocean_field.hindcast_data_source,
                                 solar_source=self.solar_field.hindcast_data_source,
                                 geographic_coordinate_system=geographic_coordinate_system)
        self.geographic_coordinate_system = geographic_coordinate_system

        # Initialize variables for holding the platform and state
        self.initial_state, self.state_trajectory, self.action_trajectory = [None]*3

    def reset(self, platform_state: PlatformState) -> ArenaObservation:
        """Resets the arena.
    Args:
        platform_state
    Returns:
      The first observation from the newly reset simulator
    """
        self.initial_state = platform_state
        self.platform.set_state(self.initial_state)
        self.platform.initialize_dynamics(self.initial_state)
        # TODO: Shall we keep those trajectories as np arrays or log them also as objects which we can transfer back
        # and forth to numpy arrays when we want to?
        self.state_trajectory = np.array([[platform_state.lon.deg, platform_state.lat.deg]])
        self.action_trajectory = np.zeros(shape=(0, 2))
        return ArenaObservation(platform_state=platform_state)

    def step(self, action: PlatformAction) -> ArenaObservation:
        """Simulates the effects of choosing the given action in the system.
    Args:
        action: The action to take in the simulator.
    Returns:
        Arena Observation including platform state, true current at platform, forecasts
    """
        state = self.platform.simulate_step(action)
        state_numpy = np.expand_dims(np.array([state.lon.deg, state.lat.deg]).squeeze(), axis=0)
        self.state_trajectory = np.append(self.state_trajectory, state_numpy, axis=0)
        action_numpy = np.expand_dims(np.array([action.magnitude, action.direction]).squeeze(), axis=0)
        self.action_trajectory = np.append(self.action_trajectory, action_numpy, axis=0)
        return ArenaObservation(platform_state=state)

    def do_nice_plot(self, x_T):
        data_store = simulation_utils.copernicusmarine_datastore('cmems_mod_glo_phy_anfc_merged-uv_PT1H-i', 'mmariuswiggert', 'tamku3-qetroR-guwneq')
        DS_currents = xr.open_dataset(data_store)[['uo', 'vo']].isel(depth=0)
        file_dicts = {'data_source_type': 'cop_opendap', 'content': DS_currents, 'grid_dict': Problem.derive_grid_dict_from_xarray(DS_currents)}
        plotting_utils.plot_2D_traj_over_currents(
            x_traj=self.state_trajectory.T,
            deg_around_x0_xT_box=0.5,
            x_T=x_T,
            x_T_radius=0.1,
            time=self.initial_state.date_time.timestamp(),
            file_dicts=file_dicts,
            ctrl_seq=self.action_trajectory.T,
            u_max=0.1
        )

