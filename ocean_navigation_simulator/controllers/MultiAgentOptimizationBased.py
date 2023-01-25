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
import math
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
        x_t = np.array(self.observation.platform_state.to_spatio_temporal_point())
        T, u, x = self.run_optimization(
            x_t=x_t,
            u_hj=np.array(hj_optimal_ctrl),
        )

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
        x = [opti.variable(self.nb_platforms, 2) for _ in range(N + 1)]
        # Decision variables for state trajetcory
        u = [
            opti.variable(self.nb_platforms, 2) for _ in range(N)
        ]  # Decision variables for input trajectory

        # Parameters (not optimized over)
        x_start = opti.parameter(self.nb_platforms, 2)

        # optimizer objective
        opti.minimize(ca.dot(u[0] - u_hj, u[0] - u_hj))

        # init the dynamics constraints
        F_dyn = self.dynamics(opti=opti)
        # F_dyn = ca.Function(
        #     "f",
        #     [x[0], u[0], t],
        #     [
        #         ca.horzcat(
        #             u[0][:, 0]
        #             + self.ocean_source.u_curr_func(ca.horzcat(t, x[0][:, 0], x[0][:, 1]).T).T,
        #             u[0][:, 1]
        #             + self.ocean_source.v_curr_func(ca.horzcat(t, x[0][:, 0], x[0][:, 1]).T).T,
        #         )
        #     ],
        #     ["x", "u", "t"],
        #     ["x_dot"],
        # )
        # x_state = opti.variable(self.nb_platforms, 2)
        # u_ctrl = opti.variable(self.nb_platforms, 2)
        # F_dyn = ca.Function(
        #     "f",
        #     [x_state, u_ctrl, t],
        #     [
        #         ca.horzcat(
        #             u_ctrl[:, 0],
        #             # + self.ocean_source.u_curr_func(
        #             #     ca.horzcat(t, x_state[:, 0], x_state[:, 1]).T
        #             # ).T,
        #             u_ctrl[:, 1]
        #             # + self.ocean_source.v_curr_func(
        #             #     ca.horzcat(t, x_state[:, 0], x_state[:, 1]).T
        #             # ).T,
        #         )
        #     ],
        #     ["x", "u", "t"],
        #     ["x_dot"],
        # )

        # add the dynamics constraints
        dt = T / N
        for k in range(N):
            # calculate time in POSIX seconds
            time = x_t[:, 2] + k * dt
            # explicit forward euler version
            x_next = x[k] + dt * F_dyn(x=x[k], u=u[k], t=time)["x_dot"]
            opti.subject_to(x[k + 1] == x_next)

        # start state & goal constraint
        opti.subject_to(x[0] == x_start)

        # control constraints
        for k in range(N):
            opti.subject_to(u[k][:, 0] ** 2 + u[k][:, 1] ** 2 <= 1)

        # State constraints to the map boundaries
        # opti.subject_to(opti.bounded(self.grids_dict['x_grid'].min(),
        #                              x[0, :], self.grids_dict['x_grid'].max()))
        # opti.subject_to(opti.bounded(self.grids_dict['y_grid'].min(),
        #                              x[1, :], self.grids_dict['y_grid'].max()))

        # battery constraint
        # opti.subject_to(opti.bounded(0.1, x[2, :], 1.))

        # Set the values for the optimization problem
        opti.set_value(x_start, x_t[:, :2])

        # Optional: initialize variables for the optimizer
        # opti.set_initial(x, x_init)
        # opti.set_initial(u, u_init)

        opti.solver("ipopt")
        sol = opti.solve()

        # extract the time vector and control signal
        T = sol.value(T)
        u = [sol.value(_) for _ in u]
        x = [sol.value(_) for _ in x]

        return T, u, x

    def dynamics(self, opti):
        # create the dynamics function (note here in form of u_x and u_y)
        x_state = ca.MX.sym("state", self.nb_platforms, 2)  # opti.variable(self.nb_platforms, 2)
        # sym_lon_degree = opti.variable(self.nb_platforms, 1)  # in deg or m
        # sym_lat_degree = opti.variable(self.nb_platforms, 1)  # in deg or m
        sym_time = ca.MX.sym("time", self.nb_platforms, 1)  # in posix
        u = ca.MX.sym("ctrl_input", self.nb_platforms, 2)  # opti.variable(self.nb_platforms, 2)
        # For interpolation: need to pass a matrix (time, lat,lon) x nb_platforms (i.e. platforms as columns and not as rows)
        u_curr = self.ocean_source.u_curr_func(
            ca.horzcat(sym_time, x_state[:, 1], x_state[:, 0]).T
        ).T  # retranspose it back to a vector where platforms are rows
        v_curr = self.ocean_source.v_curr_func(
            ca.horzcat(sym_time, x_state[:, 1], x_state[:, 0]).T
        ).T
        lon_delta_meters_per_s = u[:, 0] + u_curr
        lat_delta_meters_per_s = u[:, 1] + v_curr
        lon_delta_deg_per_s = (
            180 * lon_delta_meters_per_s / math.pi / 6371000 / ca.cos(math.pi * x_state[:, 1] / 180)
        )
        lat_delta_deg_per_s = 180 * lat_delta_meters_per_s / math.pi / 6371000
        F_next = ca.Function(
            "F_x_next",
            [x_state, u, sym_time],
            [ca.horzcat(lon_delta_deg_per_s, lat_delta_deg_per_s)],
            ["x", "u", "t"],
            ["x_dot"],
        )
        return F_next


#     def run_optimization(self, x_t, u_hj):
#         """Set-up the ipopt problem and solve it.
#         Input:
#             x_t     the starting point of the optimization
#             u_hj    the ideal control input for reachability: [u_x u_y] x #platforms
#         Returns:
#             - T     total time from start to end
#             - u     array of controls over time (u_x, u_y)
#             - x     state vector lat,lon over time
#             - dt    time between to points/control times
#         """
#         # implemented by the child classes
#         pass


# class Optimizer(MultiAgentOptim):
#     def __init__(self, observation, param_dict, platform_dict):
#         # initialize superclass
#         super().__init__(self, observation, param_dict, platform_dict)
