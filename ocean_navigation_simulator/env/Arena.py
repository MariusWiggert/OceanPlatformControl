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
from ocean_navigation_simulator.env.data_sources.OceanCurrentSource.OceanCurrentVector import OceanCurrentVector
import ocean_navigation_simulator.env.utils.units as units


@dataclasses.dataclass
class ArenaObservation:
    """
    Specifies an observation from the simulator.
    This differs from SimulatorState in that the observations are not
    ground truth state, and are instead noisy observations from the
    environment.
    """
    platform_state: PlatformState                       # position, time, battery
    true_current_at_state: OceanCurrentVector           # measured current at platform_state
    forecasted_current_at_state: OceanCurrentVector     # forecasted current at platform_state


class Arena:
    """A OceanPlatformArena in which an ocean platform moves through a current field."""

    # TODO: where do we do the reset? I guess for us reset mostly would mean new start and goal position?
    # TODO: not sure what that should be for us, decide where to put the feature constructor
    def __init__(self, sim_cache_dict: Dict, platform_dict: Dict, ocean_dict: Dict,
                 solar_dict: Optional[Dict] = None, seaweed_dict: Optional[Dict] = None,
                 use_geographic_coordinate_system: Optional[bool] = True):
        """OceanPlatformArena constructor.
    Args:
        sim_cache_dict:
        platform_dict:
        ocean_dict:
        solar_dict:
        seaweed_dict:
        geographic_coordinate_system
    Optional Args:
        geographic_coordinate_system: If True we use the Geographic coordinate system in lat, lon degree, if false the spatial system is in meters in x, y.
    """
        # Initialize the Data Fields from the respective dictionaries
        self.ocean_field = OceanCurrentField(sim_cache_dict=sim_cache_dict,
                                             hindcast_source_dict=ocean_dict['hindcast'],
                                             forecast_source_dict=ocean_dict['forecast'],
                                             use_geographic_coordinate_system=use_geographic_coordinate_system)
        if solar_dict is not None and solar_dict['hindcast'] is not None:
            self.solar_field = SolarIrradianceField(sim_cache_dict=sim_cache_dict,
                                                    hindcast_source_dict=solar_dict['hindcast'],
                                                    forecast_source_dict=solar_dict['forecast'],
                                                    use_geographic_coordinate_system=use_geographic_coordinate_system)
        else:
            self.solar_field = None

        if seaweed_dict is not None and seaweed_dict['hindcast'] is not None:
            # For initializing the SeaweedGrowth Field we need to supply the respective SolarIrradianceSources
            seaweed_dict['hindcast']['source_settings']['solar_source'] = self.solar_field.hindcast_data_source
            if seaweed_dict['forecast'] is not None:
                seaweed_dict['forecast']['source_settings']['solar_source'] = self.solar_field.hindcast_data_source
            self.seaweed_field = SeaweedGrowthField(sim_cache_dict=sim_cache_dict,
                                                    hindcast_source_dict=seaweed_dict['hindcast'],
                                                    forecast_source_dict=seaweed_dict['forecast'],
                                                    use_geographic_coordinate_system=use_geographic_coordinate_system)
        else:
            self.seaweed_field = None

        # Initialize the Platform Object from the dictionary
        self.platform = Platform(platform_dict=platform_dict,
                                 ocean_source=self.ocean_field.hindcast_data_source,
                                 solar_source=self.solar_field.hindcast_data_source if self.solar_field is not None else None,
                                 seaweed_source=self.seaweed_field.hindcast_data_source if self.seaweed_field is not None else None)

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
        self.state_trajectory = np.expand_dims(np.array(platform_state).squeeze(), axis=0)
        self.action_trajectory = np.zeros(shape=(0, 2))
        return ArenaObservation(platform_state=platform_state,
                                true_current_at_state=self.ocean_field.get_ground_truth(
                                    self.initial_state.to_spatio_temporal_point()),
                                forecasted_current_at_state=self.ocean_field.get_forecast(
                                    self.initial_state.to_spatio_temporal_point())
                                )

    def step(self, action: PlatformAction) -> ArenaObservation:
        """Simulates the effects of choosing the given action in the system.
    Args:
        action: The action to take in the simulator.
    Returns:
        Arena Observation including platform state, true current at platform, forecasts
    """
        state = self.platform.simulate_step(action)

        self.state_trajectory = np.append(self.state_trajectory, np.expand_dims(np.array(state).squeeze(), axis=0), axis=0)
        self.action_trajectory = np.append(self.action_trajectory, np.expand_dims(np.array(action).squeeze(), axis=0), axis=0)

        return ArenaObservation(platform_state=state,
                                true_current_at_state=self.ocean_field.get_ground_truth(
                                    state.to_spatio_temporal_point()),
                                forecasted_current_at_state=self.ocean_field.get_forecast(
                                    state.to_spatio_temporal_point()))

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

