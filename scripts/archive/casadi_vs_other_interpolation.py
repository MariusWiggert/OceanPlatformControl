import numpy as np

from ocean_navigation_simulator.data_sources.OceanCurrentSource.OceanCurrentSource import (
    ForecastFileSource,
    HindcastFileSource,
    HindcastOpendapSource,
)

# source = ForecastFileSource(source_type='string', config_dict={'folder':"data/cop_fmrc/"})
# source = HindcastFileSource(source_type='string', config_dict={'folder':"data/single_day_hindcasts/"})
source = HindcastFileSource(source_type="string", config_dict={"folder": "data/hindcast_test/"})

#%%
import datetime

t_0 = datetime.datetime(2021, 11, 23, 12, 10, 10, tzinfo=datetime.timezone.utc)
# t_0 = datetime.datetime(2022, 4, 4, 23, 30, tzinfo=datetime.timezone.utc)
t_interval = [t_0, t_0 + datetime.timedelta(days=1)]
x_interval = [-82, -79]
y_interval = [-22, -25]
x_0 = [-81.5, 23.5, 1]  # lon, lat, battery
x_T = [-80.4, 24.2]

deg_around_x_t: 2
t_horizon_sim: 10
from datetime import datetime, timezone

#%% get the data
import ocean_navigation_simulator.utils as utils

# Step 0: set up time and lat/lon bounds for data sub-setting
x_t = sim.cur_state.flatten()
t_upper = min(
    x_t[3] + sim.sim_settings["hours_to_sim_timescale"] * 24 * sim.sim_settings["t_horizon_sim"],
    sim.problem.hindcast_data_source["grid_dict"]["t_range"][1].timestamp(),
)

t_interval = [
    datetime.fromtimestamp(x_t[3], tz=timezone.utc),
    datetime.fromtimestamp(t_upper, tz=timezone.utc),
]
lon_interval = [
    x_t[0] - sim.sim_settings["deg_around_x_t"],
    x_t[0] + sim.sim_settings["deg_around_x_t"],
]
lat_interval = [
    x_t[1] - sim.sim_settings["deg_around_x_t"],
    x_t[1] + sim.sim_settings["deg_around_x_t"],
]

# Step 2.1: read the relevant subset of data
sim.grids_dict, u_data, v_data = utils.simulation_utils.get_current_data_subset(
    t_interval, lat_interval, lon_interval, sim.problem.hindcast_data_source
)
#%% create an interpolation function
from scipy import ndimage

start_time = time.time()
10 + 0.5 * ndimage.map_coordinates(u_data, [[0.5], [0.5], [0.5]], order=1)
10 + 0.5 * ndimage.map_coordinates(v_data, [[0.5], [0.5], [0.5]], order=1)
print((time.time() - start_time) * 1000)
# -> 0.2 - 0.6 ms! (Plus some compute for the grid stuff on-top but could be negligible
#%% Jax experiment:
state = x_0[:2]
time_input = sim.grids_dict["t_grid"][10] + 40
grid_0 = sim.grids_dict["x_grid"]
grid_1 = sim.grids_dict["y_grid"]
grid_2 = sim.grids_dict["t_grid"]
field_matrix = u_data
#%%
# Step 2.2: get the current interpolation functions
u_curr_func, v_curr_func = utils.simulation_utils.get_interpolation_func(
    sim.grids_dict, u_data, v_data, type=sim.sim_settings["int_pol_type"]
)
#%%
start_time = time.time()
10 + 0.4 * u_curr_func([time_input, x_0[0], x_0[1]]).__float__()
10 + 0.4 * v_curr_func([time_input, x_0[0], x_0[1]]).__float__()
print((time.time() - start_time) * 1000)
# 0.2-0.3ms consistently! That is the fastest!
#%%
import time

start_time = time.time()
lin_interpo_3D_fields(state, time_input, u_data, grid_0, grid_1, grid_2)
lin_interpo_3D_fields(state, time_input, v_data, grid_0, grid_1, grid_2)
print((time.time() - start_time) * 1000)
# 0.2 - 0.5 ms after compilation. Probably even faster when run on GPUs, so might be worth it...
#%%
start_time = time.time()
utils.solar_rad(time_input, x_0[0], x_0[1])
print((time.time() - start_time) * 1000)
# this takes 2.9ms. Weird, because it's actually called inside the integration loop where it then doesn't take that long.
# => that speaks for sticking with the full casadi composed integration.
# OR more general: we'd potentially like this to also be a linear interpolation when we create a 3D array from it.
# Then we can just add in actual solar fields too if we want to
#%%
import jax.numpy as jnp
import jax.scipy.ndimage
import numpy as np
from jax import jit
from scipy import interpolate


@jit
def lin_interpo_3D_fields(state, time, field_matrix, grid_0, grid_1, grid_2):
    """3D interpolation is performed using the jax implemented map_coordinates function.
    For that, first the query point in state (lat, lon) and time is transformed to float indices along each
    of the input grids (idx_x, idx_y, idx_t).

    Note: out of bound checking is not provided we extrapolate with the end values.
    If clauses are not jit-able hence we would lose speed if we checked.
    """
    # translate point to array of idx
    idx_point = jnp.array(
        [
            jnp.interp(state[0], grid_0, np.arange(len(grid_0))),
            jnp.interp(state[1], grid_1, np.arange(len(grid_1))),
            jnp.interp(time, grid_2, np.arange(len(grid_2))),
        ]
    ).reshape(-1, 1)

    # apply map_coordinates
    field_val = jax.scipy.ndimage.map_coordinates(field_matrix, idx_point, order=1)
    return field_val[0]


#%%

#%%
# Step 1: define variables
x_sym_1 = ca.MX.sym("x1")  # lon
x_sym_2 = ca.MX.sym("x2")  # lat
x_sym_3 = ca.MX.sym("x3")  # battery
x_sym_4 = ca.MX.sym("t")  # time
x_sym = ca.vertcat(x_sym_1, x_sym_2, x_sym_3, x_sym_4)

u_sim_1 = ca.MX.sym("u_1")  # thrust magnitude in [0,1]
u_sim_2 = ca.MX.sym("u_2")  # header in radians
u_sym = ca.vertcat(u_sim_1, u_sim_2)

# Step 2.1: read the relevant subset of data
self.grids_dict, u_data, v_data = simulation_utils.get_current_data_subset(
    t_interval, lat_interval, lon_interval, self.problem.hindcast_data_source
)

# Step 2.2: get the current interpolation functions
u_curr_func, v_curr_func = simulation_utils.get_interpolation_func(
    self.grids_dict, u_data, v_data, type=self.sim_settings["int_pol_type"]
)

# Step 2.1: relative charging of the battery depends on unix time, lat, lon
charge = self.problem.dyn_dict["solar_factor"] * solar_rad(x_sym[3], x_sym[1], x_sym[0])

# Step 3.1: create the x_dot dynamics function
x_dot_func = ca.Function(
    "f_x_dot",
    [x_sym, u_sym],
    [
        ca.vertcat(
            (
                ca.cos(u_sym[1]) * u_sym[0] * self.problem.dyn_dict["u_max"]
                + u_curr_func(ca.vertcat(x_sym[3], x_sym[1], x_sym[0]))
            )
            / self.sim_settings["conv_m_to_deg"],
            (
                ca.sin(u_sym[1]) * u_sym[0] * self.problem.dyn_dict["u_max"]
                + v_curr_func(ca.vertcat(x_sym[3], x_sym[1], x_sym[0]))
            )
            / self.sim_settings["conv_m_to_deg"],
            charge
            - self.problem.dyn_dict["energy"] * (self.problem.dyn_dict["u_max"] * u_sym[0]) ** 3,
            1,
        )
    ],
    ["x", "u"],
    ["x_dot"],
)

# Step 3.2: create an integrator out of it
if self.sim_settings["sim_integration"] == "rk":
    dae = {"x": x_sym, "p": u_sym, "ode": x_dot_func(x_sym, u_sym)}
    integ = ca.integrator("F_int", "rk", dae, {"tf": self.sim_settings["dt"]})
    # Simplify API to (x,u)->(x_next)
    F_x_next = ca.Function(
        "F_x_next", [x_sym, u_sym], [integ(x0=x_sym, p=u_sym)["xf"]], ["x", "u"], ["x_next"]
    )

elif self.sim_settings["sim_integration"] == "ef":
    F_x_next = ca.Function(
        "F_x_next",
        [x_sym, u_sym],
        [x_sym + self.sim_settings["dt"] * x_dot_func(x_sym, u_sym)],
        ["x", "u"],
        ["x_next"],
    )
else:
    raise ValueError("sim_integration: only RK4 (rk) and forward euler (ef) implemented")

# set the class variable
self.F_x_next = F_x_next
