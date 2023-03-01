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
from ocean_navigation_simulator.utils import units


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
        self.r_deg = units.Distance(m=self.r)
        self.optim_horizon = param_dict["optim_horizon"]
        self.list_all_platforms = list(range(self.adjacency_mat.shape[0]))
        self.nb_platforms = len(self.list_all_platforms)

    def get_next_control_for_all_pltf(self, hj_optimal_ctrl, hj_horizon_full: bool):
        # Step 1: read the relevant subset of data
        self.ocean_source = self.observation.forecast_data_source.forecast_data_source
        x_t = np.array(self.observation.platform_state.to_spatio_temporal_point())
        if hj_horizon_full:
            T, u, x, time_solver = self.run_optimization_over_full_hj(
                x_t=x_t,
                u_hj=np.array(hj_optimal_ctrl),
            )
        else:
            T, u, x, time_solver = self.run_optimization_safety_filter(
                x_t=x_t,
                u_hj=np.array(hj_optimal_ctrl),
            )
        first_actions = u[0]
        return [
            PlatformAction.from_xy_propulsion(first_actions[i, 0], first_actions[i, 1])
            for i in range(self.nb_platforms)
        ], time_solver

    def run_optimization_over_full_hj(self, x_t, u_hj):
        """Set-up the ipopt problem and solve it.
            receive x_t as a np.array spatioTemporal point, states as columns and platforms as rows
        Input:
            x_t     the starting point of the optimization
            u_hj    the ideal control input for reachability: [u_x u_y] x #platforms
        Returns:
            - T     total time from start to end
            - u     array of controls over time (u_x, u_y)
            - x     state vector lat,lon over time
            - dt    time between to points/control times
        """
        # create optimization problem
        opti = ca.Opti()

        # declare fixed End time and variable t for indexing to variable current
        dt = self.dt_in_s
        H = self.optim_horizon
        T = dt * H  # horizon in seconds

        # Objective scaling: prioritise collision avoidance and keeping communication
        max_mag_to_u_hj = sum(
            [
                np.linalg.norm(2 * u_hj_pred_k, "fro") ** 2 / self.nb_platforms
                for u_hj_pred_k in u_hj
            ]
        )
        scaling_pot_func = max_mag_to_u_hj * self.param_dict["scaling_pot_function"]

        # Declare decision variables
        x = [opti.variable(self.nb_platforms, 2) for _ in range(H)]
        # Decision variables for state trajetcory
        u = [
            opti.variable(self.nb_platforms, 2) for _ in range(H - 1)
        ]  # Decision variables for input trajectory

        # Parameters (not optimized over)
        x_start = opti.parameter(self.nb_platforms, 2)

        # init the dynamics constraints
        F_dyn = self.dynamics(opti=opti)
        objective = []
        for k in range(H - 1):
            # ----- Add the dynamics equality constraints ----- #
            # Calculate time in POSIX seconds
            time = x_t[:, 2] + k * dt
            # Explicit forward euler version
            x_next = x[k] + dt * F_dyn(x=x[k], u=u[k], t=time)["x_dot"]
            opti.subject_to(x[k + 1] == x_next)

            # ----- Add state constraints to the map boundaries ----- #
            opti.subject_to(
                opti.bounded(
                    min(self.ocean_source.grid_dict["x_range"]),
                    x[k][:, 0],
                    max(self.ocean_source.grid_dict["x_range"]),
                )
            )
            opti.subject_to(
                opti.bounded(
                    min(self.ocean_source.grid_dict["y_range"]),
                    x[k][:, 1],
                    max(self.ocean_source.grid_dict["y_range"]),
                )
            )
            # ----- Input constraints ----- #
            opti.subject_to(u[k][:, 0] ** 2 + u[k][:, 1] ** 2 <= 1)

            # ----- Objective Function ----- #
            # Add potential function terms over neighbors
            for platform in self.list_all_platforms:
                # neighbors_idx = [idx for idx in self.list_all_platforms if idx != platform]
                # get neighbor set of current platform
                neighbors_idx = np.argwhere(self.binary_adjacency[platform, :] > 0).flatten()
                # neighbors_idx = np.argwhere(
                #     (self.adjacency_mat[platform, :] < self.r_deg.m * 1.2)
                #     & (self.adjacency_mat[platform, :] > 0)
                # ).flatten()
                potential = scaling_pot_func * self.pot_func_pltf_i(
                    x_k=x[k], platform_id=platform, platform_neighbors_id=neighbors_idx
                )
                objective.append(potential)

            # Term to penalize deviation from optimal HJ_reachability control
            objective.append(ca.dot(u[k] - u_hj[k], u[k] - u_hj[k]))

        # ----- Terminal state constraint (at Horizon H) ----- #
        opti.subject_to(
            opti.bounded(
                min(self.ocean_source.grid_dict["x_range"]),
                x[H - 1][:, 0],
                max(self.ocean_source.grid_dict["x_range"]),
            )
        )
        opti.subject_to(
            opti.bounded(
                min(self.ocean_source.grid_dict["y_range"]),
                x[H - 1][:, 1],
                max(self.ocean_source.grid_dict["y_range"]),
            )
        )

        # optimizer objective
        opti.minimize(sum(objective))

        # start state & goal constraint
        opti.subject_to(x[0] == x_start)

        # Set the values for the optimization problem
        opti.set_value(x_start, x_t[:, :2])

        # Initialize variables for the optimizer: avoids unfeasibility issues
        for k in range(H):
            opti.set_initial(x[k], x_t[:, :2])
            if k < H - 1:  # control input seq length of length H-1
                opti.set_initial(u[k], u_hj[k])

        opti.solver("ipopt", {"print_time": True})
        try:
            sol = opti.solve()
            T = sol.value(T)
            u = [sol.value(_) for _ in u]
            x = [sol.value(_) for _ in x]
            stats = opti.stats()
        except BaseException:
            stats = opti.stats()
            if stats["return_status"] == "Maximum_Iterations_Exceeded":
                T = opti.debug.value(T)
                u = [opti.debug.value(_) for _ in u]
                x = [opti.debug.value(_) for _ in x]
            else:
                raise Exception("problem infeasible")

        time_total = opti.stats()["t_proc_total"]

        return T, u, x, time_total

    def run_optimization_safety_filter(self, x_t, u_hj):
        """Set-up the ipopt problem and solve it.
            receive x_t as a np.array spatioTemporal point, states as columns and platforms as rows
        Input:
            x_t     the starting point of the optimization
            u_hj    the ideal control input for reachability: [u_x u_y] x #platforms
        Returns:
            - T     total time from start to end
            - u     array of controls over time (u_x, u_y)
            - x     state vector lat,lon over time
            - dt    time between to points/control times
        """
        # create optimization problem
        opti = ca.Opti()

        # declare fixed End time and variable t for indexing to variable current
        dt = self.dt_in_s
        H = self.optim_horizon
        T = dt * H  # horizon in seconds

        # Objective scaling: prioritise collision avoidance and keeping communication
        max_mag_to_u_hj = np.linalg.norm(2 * u_hj, "fro") ** 2 / self.nb_platforms
        scaling_pot_func = max_mag_to_u_hj * self.param_dict["scaling_pot_function"]

        # Declare decision variables
        x = [opti.variable(self.nb_platforms, 2) for _ in range(H)]
        # Decision variables for state trajetcory
        u = [
            opti.variable(self.nb_platforms, 2) for _ in range(H - 1)
        ]  # Decision variables for input trajectory

        # Parameters (not optimized over)
        x_start = opti.parameter(self.nb_platforms, 2)

        # init the dynamics constraints
        F_dyn = self.dynamics(opti=opti)
        objective = []
        for k in range(H - 1):
            # ----- Add the dynamics equality constraints ----- #
            # Calculate time in POSIX seconds
            time = x_t[:, 2] + k * dt
            # Explicit forward euler version
            x_next = x[k] + dt * F_dyn(x=x[k], u=u[k], t=time)["x_dot"]
            opti.subject_to(x[k + 1] == x_next)

            # ----- Add state constraints to the map boundaries ----- #
            opti.subject_to(
                opti.bounded(
                    min(self.ocean_source.grid_dict["x_range"]),
                    x[k][:, 0],
                    max(self.ocean_source.grid_dict["x_range"]),
                )
            )
            opti.subject_to(
                opti.bounded(
                    min(self.ocean_source.grid_dict["y_range"]),
                    x[k][:, 1],
                    max(self.ocean_source.grid_dict["y_range"]),
                )
            )
            # ----- Input constraints ----- #
            opti.subject_to(u[k][:, 0] ** 2 + u[k][:, 1] ** 2 <= 1)

            # ----- Objective Function ----- #
            # Add potential function terms over neighbors
            for platform in self.list_all_platforms:
                # neighbors_idx = [idx for idx in self.list_all_platforms if idx != platform]
                # get neighbor set of current platform
                neighbors_idx = np.argwhere(self.binary_adjacency[platform, :] > 0).flatten()
                # neighbors_idx = np.argwhere(
                #     (self.adjacency_mat[platform, :] < self.r_deg.m * 1.2)
                #     & (self.adjacency_mat[platform, :] > 0)
                # ).flatten()
                potential = scaling_pot_func * self.pot_func_pltf_i(
                    x_k=x[k], platform_id=platform, platform_neighbors_id=neighbors_idx
                )
                objective.append(potential)

        # Term to penalize deviation from optimal HJ_reachability control
        objective.append(ca.dot(u[0] - u_hj, u[0] - u_hj))

        # ----- Terminal state constraint (at Horizon H) ----- #
        opti.subject_to(
            opti.bounded(
                min(self.ocean_source.grid_dict["x_range"]),
                x[H - 1][:, 0],
                max(self.ocean_source.grid_dict["x_range"]),
            )
        )
        opti.subject_to(
            opti.bounded(
                min(self.ocean_source.grid_dict["y_range"]),
                x[H - 1][:, 1],
                max(self.ocean_source.grid_dict["y_range"]),
            )
        )

        # optimizer objective
        opti.minimize(sum(objective))

        # start state & goal constraint
        opti.subject_to(x[0] == x_start)

        # Set the values for the optimization problem
        opti.set_value(x_start, x_t[:, :2])

        # Initialize variables for the optimizer: avoids unfeasibility issues
        for k in range(H):
            opti.set_initial(x[k], x_t[:, :2])

        opti.set_initial(u[0], u_hj)

        opti.solver("ipopt", {"print_time": True})
        try:
            sol = opti.solve()
            T = sol.value(T)
            u = [sol.value(_) for _ in u]
            x = [sol.value(_) for _ in x]
            stats = opti.stats()
        except BaseException:
            stats = opti.stats()
            if stats["return_status"] == "Maximum_Iterations_Exceeded":
                T = opti.debug.value(T)
                u = [opti.debug.value(_) for _ in u]
                x = [opti.debug.value(_) for _ in x]
            else:
                raise Exception("problem infeasible")

        time_total = opti.stats()["t_proc_total"]

        return T, u, x, time_total

    def dynamics(self, opti):
        # create the dynamics function (note here in form of u_x and u_y)
        # x_state = opti.variable(self.nb_platforms, 2)
        x_state = ca.MX.sym("state", self.nb_platforms, 2)
        # sym_lon_degree = opti.variable(self.nb_platforms, 1)  # in deg or m
        # sym_lat_degree = opti.variable(self.nb_platforms, 1)  # in deg or m
        sym_time = ca.MX.sym("time", self.nb_platforms, 1)  # in posix
        # u = opti.variable(self.nb_platforms, 2)
        u = ca.MX.sym("ctrl_input", self.nb_platforms, 2)  #
        # For interpolation: need to pass a matrix (time, lat,lon) x nb_platforms (i.e. platforms as columns and not as rows)
        u_curr = self.ocean_source.u_curr_func(
            ca.horzcat(sym_time, x_state[:, 1], x_state[:, 0]).T
        ).T  # retranspose it back to a vector where platforms are rows
        v_curr = self.ocean_source.v_curr_func(
            ca.horzcat(sym_time, x_state[:, 1], x_state[:, 0]).T
        ).T
        lon_delta_meters_per_s = u[:, 0] * self.u_max_mps + u_curr
        lat_delta_meters_per_s = u[:, 1] * self.u_max_mps + v_curr
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

    def pot_func_pltf_i(self, x_k, platform_id, platform_neighbors_id):
        ideal_dist = self.r_deg.km / 2
        potentials = [
            (
                1
                / ideal_dist
                * units.Distance(deg=self.norm_2(x_k[platform_id, :], x_k[neighbor_id, :])).km
                # ca.norm_2(x_k[platform_id, :] - x_k[neighbor_id, :])).km
                - 1
            )
            ** 6
            for neighbor_id in platform_neighbors_id
        ]

        return sum(potentials)

    def norm_2(self, x_i, x_j):
        return ca.sqrt(
            ((x_i[0] - x_j[0]) * ca.cos(x_i[1]) * np.pi / 180) ** 2 + (x_i[1] - x_j[1]) ** 2
        )
