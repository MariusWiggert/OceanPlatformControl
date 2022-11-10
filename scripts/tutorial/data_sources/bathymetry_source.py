import datetime

from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.utils import units

from ocean_navigation_simulator.data_sources.BathymetryField import BathymetryField

# Initialize bathymetry source
casadi_cache_dict = {"deg_around_x_t": 1}  # No time needed as map is static
bathymetry_source_dict = {
    "field": "Bathymetry",
    "source": "gebco",
    "source_settings": {"filepath": "./data/bathymetry/bathymetry_global_res_0.083_0.083_max.nc"},
}
# TODO: possibly instead of source_dict use "bathymetry_source_dict"
bathymetry_field = BathymetryField(
    casadi_cache_dict=casadi_cache_dict, source_dict=bathymetry_source_dict
)

#%% Plot bathymetry over full field
# TODO: possibly need time as required by higher level function
bathymetry_field.data_source.plot_bathymetry()

# TODO: implent stuff after line 54 of seaweed_source.py
