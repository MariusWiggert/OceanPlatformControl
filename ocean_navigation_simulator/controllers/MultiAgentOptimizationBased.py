"""
Class implementing optimization based control for the multi-agent network of 
platforms. Uses the HJ time-optimal control input as the navigation function
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
import casadi as ca
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.integrate as integrate

from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.Platform import PlatformAction


class MultiAgentOptim:
    def __init__(
        self,
        observation: ArenaObservation,
        param_dict: dict,
        platform_dict: dict,
    ):
        self.param_dict = param_dict
        self.G_proximity = observation.graph_obs.G_communication
        self.observation = observation
        adjacency_communication = observation.graph_obs.adjacency_matrix_in_unit(
            unit=self.param_dict["unit"], graph_type="communication"
        )  # get adjacency for communication graph
        self.binary_adjacency = np.where(
            adjacency_communication > 0, True, False
        )  # indicator values
        self.adjacency_mat = observation.graph_obs.adjacency_matrix_in_unit(
            unit=self.param_dict["unit"], graph_type="complete"
        )  # complete adjacency matrix containing all distances between platforms
        self.u_max_mps = platform_dict["u_max_in_mps"]
        self.dt_in_s = platform_dict["dt_in_s"]
        self.r = self.param_dict["interaction_range"]
        self.optim_horizon = param_dict["optim_horizon"]
        self.list_all_platforms = list(range(self.adjacency_mat.shape[0]))
        self.nb_platforms = len(self.list_all_platforms)

    def potential_func(self, norm_qij: float, inside_range: bool) -> float:
        """Potential function responsible for the attraction repulsion behavior between platforms

        Args:
            norm_qij (float): the euclidean norm of the distance between platform i and j
            inside_range (bool): if the distance between i and j is within the interaction/communication range

        Returns:
            float: value of the potential function
        """
        if inside_range:
            return self.r / (norm_qij * (self.r - norm_qij))
        else:
            # return np.log(norm_qij - self.r + self.epsilon)
            return np.sqrt(norm_qij - self.r)

    def get_next_control_for_all_pltf(self, hj_optimal_ctrl):
        # Step 1: read the relevant subset of data
        self.ocean_source = self.observation.forecast_data_source
        x_t = self.observation.platform_state.to_spatio_temporal_point


class Optimizer(MultiAgentOptim):
    def __init__(self, observation, param_dict, platform_dict):
        # initialize superclass
        super().__init__(self, observation, param_dict, platform_dict)

    def run_optimization(self, x_t, u_hj):
        # receive x_t as a np.array spatioTemporal point, states as columns and platforms as rows
        # create optimization problem
        opti = ca.Opti()

        # declare fixed End time and variable t for indexing to variable current
        dt = self.dt_in_s
        N = self.optim_horizon
        T = dt * N
        t = ca.MX.sym("t", self.nb_platforms, 1)  # time symbolic variable
        # declare decision variables
        # For now 2 State system (X, Y), not capturing battery or seaweed mass
        x = opti.variable(
            2, self.nb_platforms, self.N + 1
        )  # Decision variables for state trajetcory
        u = opti.variable(2, self.nb_platforms, self.N)

        # Parameters (not optimized over)
        x_start = opti.parameter(2, self.nb_platforms)

        # optimizer objective
        opti.minimize(sum(ca.dot(u[:, :, 0] - u_hj, u[:, :, 0] - u_hj)))

        # init the dynamics constraints
        F_dyn = self.dynamics()

        # add the dynamics constraints
        dt = T / self.N
        for k in range(self.N):
            # calculate time in POSIX seconds
            time = x_t[:, 2] + k * dt
            # explicit forward euler version
            x_next = x[:, :, k] + dt * np.array(F_dyn(x=x[:, :, k].T, u=u[:, :, k].T, t=time).T)
            opti.subject_to(x[:, k + 1] == x_next)

        # start state & goal constraint
        opti.subject_to(x[:, :, 0] == x_start)

        # control constraints
        for k in range(self.N):
            opti.subject_to(u[0, :, k] ** 2 + u[1, :, k] ** 2 <= 1)

        # State constraints to the map boundaries
        # opti.subject_to(opti.bounded(self.grids_dict['x_grid'].min(),
        #                              x[0, :], self.grids_dict['x_grid'].max()))
        # opti.subject_to(opti.bounded(self.grids_dict['y_grid'].min(),
        #                              x[1, :], self.grids_dict['y_grid'].max()))

        # battery constraint
        # opti.subject_to(opti.bounded(0.1, x[2, :], 1.))

        # Set the values for the optimization problem
        opti.set_value(x_start, x_t)

        # Optional: initialize variables for the optimizer
        # opti.set_initial(x, x_init)
        # opti.set_initial(u, u_init)

        opti.solver("ipopt")
        sol = opti.solve()

        # extract the time vector and control signal
        T = sol.value(T)
        u = sol.value(u)
        x = sol.value(x)
        dt = sol.value(dt)

        return T, u, x, dt

    def dynamics(self):
        # create the dynamics function (note here in form of u_x and u_y)
        sym_lon_degree = ca.MX.sym("lon", self.nb_platforms, 1)  # in deg or m
        sym_lat_degree = ca.MX.sym("lat", self.nb_platforms, 1)  # in deg or m
        sym_time = ca.MX.sym("time", self.nb_platforms, 1)  # in posix
        sym_u = ca.MX("control", self.nb_platforms, 2)
        # For interpolation: need to pass a matrix (time, lat,lon) x nb_platforms (i.e. platforms as columns and not as rows)
        u_curr = self.ocean_source.u_curr_func(
            ca.horzcat(t, sym_lat_degree, sym_lon_degree).T
        ).T  # retranspose it back to a vector where platforms are rows
        v_curr = self.ocean_source.v_curr_func(
            ca.horzcat(sym_time, sym_lat_degree, sym_lon_degree).T
        ).T
        sym_lon_delta_meters_per_s = sym_u[0] + u_curr
        sym_lat_delta_meters_per_s = sym_u[1] + v_curr
        sym_lon_delta_deg_per_s = (
            180
            * sym_lon_delta_meters_per_s
            / math.pi
            / 6371000
            / ca.cos(math.pi * sym_lat_degree / 180)
        )
        sym_lat_delta_deg_per_s = 180 * sym_lat_delta_meters_per_s / math.pi / 6371000
        F_next = ca.Function(
            "F_x_next",
            [ca.horzcat(sym_lon_degree, sym_lat_degree), sym_u, sym_t],
            [ca.horzcat(sym_lon_delta_deg_per_s, sym_lat_delta_deg_per_s)],
            ["x", "u", "t"],
        )
        return F_next
