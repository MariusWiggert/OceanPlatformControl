"""
Class implementing optimization based control for the multi-agent network of
platforms. Uses the HJ time-optimal control input as the navigation function
"""

from datetime import datetime, timezone
from typing import Union, Dict, List, Optional, Tuple
import casadi as ca
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.integrate as integrate
import math
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.Platform import PlatformAction
from ocean_navigation_simulator.utils import units
from scipy.sparse import csgraph


class MultiAgentOptim:
    """Implementation of MPC using HJ as the optimal control input
    Deviation from this optimal control input is penalized in the
    objective. This version uses a potential function (similar to soft
    constraints) to penalize collisions and communication losses
    for agents that were connected
    """

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

    def get_next_control_for_all_pltf(
        self, hj_optimal_ctrl: Union[np.ndarray, List[np.ndarray]], hj_horizon_full: bool
    ) -> Tuple[List[PlatformAction], float]:
        """Main function to get the multi-agent MPC control

        Args:
            hj_optimal_ctrl (Union[np.ndarray, List[np.ndarray]]): HJ input from planner in the form of xy propulsion
                                                                   can be just for the next timestep or for the full
                                                                   horizon
            hj_horizon_full (bool): true to use the HJ inputs over the full horizon or false if only uses the one for
                                    the next timestep (similar to a safety filter)

        Returns:
            Tuple[List[PlatformAction], float]: Next platform actions u[0] to be applied as done usual for MPC
        """
        # Step 1: read the relevant subset of data
        self.ocean_source = self.observation.forecast_data_source.forecast_data_source
        x_t = np.array(self.observation.platform_state.to_spatio_temporal_point())
        if hj_horizon_full:  # use the HJ inputs over the whole horizon
            T, u, x, time_solver = self.run_optimization_over_full_hj(
                x_t=x_t,
                u_hj=np.array(hj_optimal_ctrl),
            )
        else:  # only use the first HJ input (at the next timestep) as for safety filters
            T, u, x, time_solver = self.run_optimization_safety_filter(
                x_t=x_t,
                u_hj=np.array(hj_optimal_ctrl),
            )
        first_actions = u[0]
        return [
            PlatformAction.from_xy_propulsion(first_actions[i, 0], first_actions[i, 1])
            for i in range(self.nb_platforms)
        ], time_solver

    def run_optimization_over_full_hj(
        self, x_t: np.ndarray, u_hj: np.ndarray
    ) -> Tuple[float, List[np.ndarray], List[np.ndarray], float]:
        """MPC Style multi-agent problem using the ipopt solver. Try staying close to ideal hj control for
            each platform by introducing quadratic term for deviations in the objective function. This is
            done here for the FULL horizon (H steaps ahead)
            To discourage collisions and communications losses, a "flocking-like" potential function is
            added in the objective to penalize deviations from the ideal comm. range for existing neighbors.
            Similar idea than having soft constraints between neighbors.

        Input:
            x_t     the starting point of the optimization
            u_hj    the ideal control input for reachability: horizon_len x [#platforms x [u_x u_y]]
        Returns:
            - T             total planning time (horizon in seconds)
            - u             list of array of controls over time: horizon_len x [#platforms x [u_x, u_y]]
            - x             list of state array over time:  horizon_len x [#platforms x [lat,lon]]
            - time_total    solver total time
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
                # get neighbor set of current platform
                neighbors_idx = np.argwhere(self.binary_adjacency[platform, :] > 0).flatten()
                # add potential function for neighbor platforms to foster staying close to ideal comm. range
                potential = scaling_pot_func * self.pot_func_pltf_i(
                    x_k=x[k], platform_id=platform, platform_neighbors_id=neighbors_idx
                )
                objective.append(potential)

            # Term to penalize deviation from optimal HJ_reachability control
            # over full horizon
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
                raise Exception(
                    f"Error during optimization: solver status = {stats['return_status']}"
                )
        time_total = opti.stats()["t_proc_total"]
        return T, u, x, time_total

    def run_optimization_safety_filter(
        self, x_t: np.ndarray, u_hj: np.ndarray
    ) -> Tuple[float, List[np.ndarray], List[np.ndarray], float]:
        """MPC Style multi-agent pr oblem using the ipopt solver. Try staying close to ideal hj control for
            each platform by introducing quadratic term for deviations in the objective function. This is
            only done here for the first input step ahead: see concept of safety filter:
            https://www.sciencedirect.com/science/article/pii/S0005109821001175
            To discourage collisions and communications losses, a "flocking-like" potential function is
            added in the objective to penalize deviations from the ideal comm. range for existing neighbors.
            Similar idea than having soft constraints between neighbors.

        Input:
            x_t     the starting point of the optimization
            u_hj    the ideal control input for reachability: horizon_len x [#platforms x [u_x u_y]]
        Returns:
            - T             total planning time (horizon in seconds)
            - u             list of array of controls over time: horizon_len x [#platforms x [u_x, u_y]]
            - x             list of state array over time:  horizon_len x [#platforms x [lat,lon]]
            - time_total    solver total time
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
                # get neighbor set of current platform
                neighbors_idx = np.argwhere(self.binary_adjacency[platform, :] > 0).flatten()
                # add potential function for neighbor platforms to foster staying close to ideal comm. range
                potential = scaling_pot_func * self.pot_func_pltf_i(
                    x_k=x[k], platform_id=platform, platform_neighbors_id=neighbors_idx
                )
                objective.append(potential)

        # Term to penalize deviation from optimal HJ_reachability control
        # ONLY first term here (k=0), not over the full horizon
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

    def dynamics(self):
        """Function to construct the F_x_next symbolic casadi function to be used for MPC"""
        x_state = ca.MX.sym("state", self.nb_platforms, 2)
        sym_time = ca.MX.sym("time", self.nb_platforms, 1)  # in posix
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

    def pot_func_pltf_i(self, x_k: ca.Opti, platform_id: int, platform_neighbors_id: np.ndarray):
        """Computes a penalization term for the objective function of the MPC scheme, inspired by
        potential functions for maintaining connectivity and avoiding collisions in flocking schemes.
        If d is the ideal distance to maintain, the potential term is normalized (1/d*norm(xi -xj) -1)^p
        where p is any pair positive integer.

        Args:
            x_k (ca.Opti): the state vector
            platform_id (int): current platform
            platform_neighbors_id (np.ndarray): neighbors of current platform

        Returns:
            _type_: Potential function for current platform taken w.r.t to all neighbor agents
        """
        ideal_dist = self.r_deg.km / 2
        potentials = [
            (
                1
                / ideal_dist
                * units.Distance(deg=self.norm_2_sq(x_k[platform_id, :], x_k[neighbor_id, :])).km
                - 1
            )
            ** 4
            for neighbor_id in platform_neighbors_id
        ]

        return sum(potentials)

    def norm_2_sq(self, x_i: ca.Opti, x_j: ca.Opti):
        """Compute the squared distance between two platforms in the ocean

        Args:
            x_i (ca.Opti): first platform
            x_j (ca.Opti): second platform
        """
        return ca.sqrt(
            ((x_i[0] - x_j[0]) * ca.cos(x_i[1]) * np.pi / 180) ** 2 + (x_i[1] - x_j[1]) ** 2
        )


##################################  BELOW  ##################################

# NOT WORKING: code for centralized MPC to use the Laplacian eigenvalue
# as a constraint for connectivity and HJ in the objective.
# Difficulty arises due to the fact that it is not possible to compute
# eigenvalues of symbolic variables for a matrix larger than 3x3 with CasADi.
# Maybe an SQP implementation could work, as shown in:

#############################################################################


# class CentralizedMultiAgentMPC:
#     def __init__(
#         self,
#         observation: ArenaObservation,
#         param_dict: dict,
#         platform_dict: dict,
#     ):
#         self.param_dict = param_dict
#         self.observation = observation
#         self.u_max_mps = platform_dict["u_max_in_mps"]
#         self.dt_in_s = platform_dict["dt_in_s"]
#         self.optim_horizon = param_dict["optim_horizon"]
#         self.adjacency_mat = observation.graph_obs.adjacency_matrix_in_unit(
#             unit=self.param_dict["unit"], graph_type="complete"
#         )  # complete adjacency matrix containing all distances between platforms
#         self.list_all_platforms = list(range(self.adjacency_mat.shape[0]))
#         self.nb_platforms = len(self.list_all_platforms)
#         self.r = self.param_dict["interaction_range"]
#         self.r_deg = units.Distance(m=self.r)

#     def get_next_control_for_all_pltf(self, hj_optimal_ctrl):
#         # Step 1: read the relevant subset of data
#         self.ocean_source = self.observation.forecast_data_source.forecast_data_source
#         x_t = np.array(self.observation.platform_state.to_spatio_temporal_point())
#         T, u, x, time_solver = self.run_optimization(
#             x_t=x_t,
#             u_hj=np.array(hj_optimal_ctrl),
#         )
#         first_actions = u[0]
#         return [
#             PlatformAction.from_xy_propulsion(first_actions[i, 0], first_actions[i, 1])
#             for i in range(self.nb_platforms)
#         ], time_solver

#     def run_optimization(self, x_t, u_hj):
#         # create optimization problem
#         opti = ca.Opti()

#         # declare fixed End time and variable t for indexing to variable current
#         dt = self.dt_in_s
#         H = self.optim_horizon
#         T = dt * H  # horizon in seconds

#         # Objective scaling: prioritise collision avoidance and keeping communication
#         max_mag_to_u_hj = np.linalg.norm(2 * u_hj, "fro") ** 2 / self.nb_platforms
#         scaling_pot_func = max_mag_to_u_hj * self.param_dict["scaling_pot_function"]
#         n_x = 2
#         n_u = 2
#         # Declare decision variables
#         # x = opti.variable(n_x * self.nb_platforms, H)
#         # # Decision variables for state trajetcory
#         # u = opti.variable(n_u * self.nb_platforms, H - 1)
#         # Declare decision variables
#         x = [opti.variable(self.nb_platforms, 2) for _ in range(H)]
#         # Decision variables for state trajetcory
#         u = [
#             opti.variable(self.nb_platforms, 2) for _ in range(H - 1)
#         ]  # Decision variables for input trajectory
#         # Decision variables for input trajectory
#         # laplacian_opti = [opti.variable(self.nb_platforms, self.nb_platforms) for _ in range(H)]
#         # laplacian_second_eig = opti.variable(H, 1)
#         gamma = opti.variable(H, 1)
#         # slack = opti.variable(self.nb_platforms, self.nb_platforms)
#         # Parameters (not optimized over)
#         x_start = opti.parameter(self.nb_platforms, 2)

#         # init the dynamics constraints
#         F_dyn = self.dynamics(opti=opti)
#         F_compute_laplacian = self.compute_graph_laplacian()
#         L = ca.SX.sym("L", self.nb_platforms, self.nb_platforms)
#         F_compute_second_eig = ca.Function("F_compute_second_eig", [L], [ca.eig_symbolic(L)[1]])
#         objective = []
#         laplacian_eval_at_t_k = []
#         second_eig_at_t_k = []
#         callback_second_eig = MyCallback("eig", self.nb_platforms)
#         for k in range(H - 1):
#             # ----- Add the dynamics equality constraints ----- #
#             # Calculate time in POSIX seconds
#             time = x_t[:, 2] + k * dt
#             # Explicit forward euler version
#             # x_k = ca.horzcat(x[: self.nb_platforms, k], x[self.nb_platforms :, k])
#             # u_k = ca.horzcat(x[: self.nb_platforms, k], x[self.nb_platforms :, k])
#             # x_k_plus_1 = x_k + dt * F_dyn(x=x_k, u=u_k, t=time)["x_dot"]
#             # x_next = ca.vertcat(x_k_plus_1[:, 0], x_k_plus_1[:, 1])

#             x_next = x[k] + dt * F_dyn(x=x[k], u=u[k], t=time)["x_dot"]
#             opti.subject_to(x[k + 1] == x_next)

#             # ----- Add state constraints to the map boundaries ----- #
#             opti.subject_to(
#                 opti.bounded(
#                     min(self.ocean_source.grid_dict["x_range"]),
#                     x[k][:, 0],
#                     max(self.ocean_source.grid_dict["x_range"]),
#                 )
#             )
#             opti.subject_to(
#                 opti.bounded(
#                     min(self.ocean_source.grid_dict["y_range"]),
#                     x[k][:, 1],
#                     max(self.ocean_source.grid_dict["y_range"]),
#                 )
#             )
#             # ----- Input constraints ----- #
#             opti.subject_to(u[k][:, 0] ** 2 + u[k][:, 1] ** 2 <= 1)

#             # Add Laplacian constraints
#             laplacian_eval_at_t_k.append(F_compute_laplacian(x_k=x[k])["laplacian"])
#             second_eig_at_t_k.append(callback_second_eig(laplacian_eval_at_t_k[k]))
#             # second_eig_at_t_k.append(F_compute_second_eig(laplacian_eval_at_t_k[k]))
#             # opti.subject_to(
#             #     laplacian_eval_at_t_k[k] + ca.DM(np.ones((self.nb_platforms, self.nb_platforms)))
#             #     > gamma[k] * ca.DM(np.eye((self.nb_platforms)))
#             # )
#             # opti.subject_to(laplacian_second_eig[k] == laplacian_eval_at_t_k[k])
#             # opti.subject_to(second_eig_at_t_k[k] >= 0.01)
#             # opti.subject_to(laplacian_second_eig[k] >= 0.1)
#             # opti.subject_to(
#             #     laplacian_opti[k] + np.ones((self.nb_platforms, self.nb_platforms))
#             #     >= gamma[k] * np.ones((self.nb_platforms, self.nb_platforms))
#             # )
#             # opti.subject_to(gamma[k] >= 0)

#             # ----- Objective Function ----- #
#             # objective.append(-gamma[k])
#             objective.append(-second_eig_at_t_k[k])

#         # terminal state constraint for laplacian
#         # laplacian_eval_k = F_compute_laplacian(x_k=x[H - 1])["laplacian"]
#         # opti.subject_to(
#         #     laplacian_opti[H - 1] + np.ones((self.nb_platforms, self.nb_platforms))
#         #     >= gamma[H - 1] * np.ones((self.nb_platforms, self.nb_platforms))
#         # )
#         # opti.subject_to(gamma[H - 1] >= 0)

#         # Term to penalize deviation from optimal HJ_reachability control
#         # objective.append(ca.dot(u[0] - u_hj, u[0] - u_hj))

#         # ----- Terminal state constraint (at Horizon H) ----- #
#         opti.subject_to(
#             opti.bounded(
#                 min(self.ocean_source.grid_dict["x_range"]),
#                 x[H - 1][:, 0],
#                 max(self.ocean_source.grid_dict["x_range"]),
#             )
#         )
#         opti.subject_to(
#             opti.bounded(
#                 min(self.ocean_source.grid_dict["y_range"]),
#                 x[H - 1][:, 1],
#                 max(self.ocean_source.grid_dict["y_range"]),
#             )
#         )

#         # optimizer objective
#         objective.append(ca.dot(u[0] - u_hj, u[0] - u_hj))
#         opti.minimize(sum(objective))

#         # start state & goal constraint
#         opti.subject_to(x[0] == x_start)

#         # Set the values for the optimization problem
#         opti.set_value(x_start, x_t[:, :2])

#         # Initialize variables for the optimizer: avoids unfeasibility issues
#         for k in range(H):
#             opti.set_initial(x[k], x_t[:, :2])

#         opti.set_initial(u[0], u_hj)

#         opti.solver("ipopt")
#         # opti.solver("ipopt", {"expand": True}, {"acceptable_constr_viol_tol": 1e-8})
#         # opti.solver("sqpmethod", {"print_time": True}) for QP
#         try:
#             sol = opti.solve()
#             T = sol.value(T)
#             u = [sol.value(_) for _ in u]
#             x = [sol.value(_) for _ in x]
#             stats = opti.stats()
#         except BaseException:
#             stats = opti.stats()
#             if stats["return_status"] == "Maximum_Iterations_Exceeded":
#                 T = opti.debug.value(T)
#                 u = [opti.debug.value(_) for _ in u]
#                 x = [opti.debug.value(_) for _ in x]
#             else:
#                 raise Exception("problem infeasible")

#         time_total = opti.stats()["t_proc_total"]

#         return T, u, x, time_total

#     def dynamics(self, opti):
#         # create the dynamics function (note here in form of u_x and u_y)
#         # x_state = opti.variable(self.nb_platforms, 2)
#         x_state = ca.MX.sym("state", self.nb_platforms, 2)
#         # sym_lon_degree = opti.variable(self.nb_platforms, 1)  # in deg or m
#         # sym_lat_degree = opti.variable(self.nb_platforms, 1)  # in deg or m
#         sym_time = ca.MX.sym("time", self.nb_platforms, 1)  # in posix
#         # u = opti.variable(self.nb_platforms, 2)
#         u = ca.MX.sym("ctrl_input", self.nb_platforms, 2)  #
#         # For interpolation: need to pass a matrix (time, lat,lon) x nb_platforms (i.e. platforms as columns and not as rows)
#         u_curr = self.ocean_source.u_curr_func(
#             ca.horzcat(sym_time, x_state[:, 1], x_state[:, 0]).T
#         ).T  # retranspose it back to a vector where platforms are rows
#         v_curr = self.ocean_source.v_curr_func(
#             ca.horzcat(sym_time, x_state[:, 1], x_state[:, 0]).T
#         ).T
#         lon_delta_meters_per_s = u[:, 0] * self.u_max_mps + u_curr
#         lat_delta_meters_per_s = u[:, 1] * self.u_max_mps + v_curr
#         lon_delta_deg_per_s = (
#             180 * lon_delta_meters_per_s / math.pi / 6371000 / ca.cos(math.pi * x_state[:, 1] / 180)
#         )
#         lat_delta_deg_per_s = 180 * lat_delta_meters_per_s / math.pi / 6371000
#         F_next = ca.Function(
#             "F_x_next",
#             [x_state, u, sym_time],
#             [ca.horzcat(lon_delta_deg_per_s, lat_delta_deg_per_s)],
#             ["x", "u", "t"],
#             ["x_dot"],
#         )
#         return F_next

#     def compute_graph_laplacian(self):
#         # Compute the graph laplacian given the input x_k that is 2-dimensional (lon, lat) in casadi, the pair x_k[i,:], x[j,:] is a pair of platforms and there is an edge if the distance between them is less than the communication range self.r_deg.km
#         # The graph laplacian is defined as L = D - A where D is the degree matrix and A is the adjacency matrix

#         # Define the adjacency matrix
#         adjacency_mat = ca.MX(self.nb_platforms, self.nb_platforms)
#         x_k = ca.MX.sym("x_k", self.nb_platforms, 2)
#         degree_mat = ca.MX(np.zeros((self.nb_platforms, self.nb_platforms)))
#         for i in range(self.nb_platforms):
#             for j in range(self.nb_platforms):
#                 if i != j:
#                     adjacency_mat[i, j] = (
#                         -0.5
#                         * ca.tanh(
#                             10 * units.Distance(deg=self.norm_2(x_k[i, :], x_k[j, :])).km
#                             - 10 * self.r_deg.km
#                         )
#                         + 0.5
#                     )
#                 else:
#                     adjacency_mat[i, j] = 0
#             degree_mat[i, i] = ca.sum2(adjacency_mat)[i]
#         laplacian = degree_mat - adjacency_mat
#         # igen_laplacian = ca.eig_symbolic(laplacian)
#         # idx = eigen_laplacian.argsort()[::-1]
#         # eigen_laplacian = eigen_laplacian[idx]
#         # Laplacian_fun = ca.Function(
#         #     "Laplacian_fun", [x_k], [eigen_laplacian[1]], ["x_k"], ["laplacian"]
#         # )
#         Laplacian_fun = ca.Function("Laplacian_fun", [x_k], [laplacian], ["x_k"], ["laplacian"])
#         return Laplacian_fun

#     def norm_2(self, x_i, x_j):
#         return ca.sqrt(
#             ((x_i[0] - x_j[0]) * ca.cos(x_i[1]) * np.pi / 180) ** 2 + (x_i[1] - x_j[1]) ** 2
#         )


# class MyCallback(ca.Callback):
#     def __init__(self, name, nb_agents, opts={}):
#         ca.Callback.__init__(self)
#         self.nb_agents = nb_agents
#         self.construct(name, opts)

#     # Number of inputs and outputs
#     def get_n_in(self):
#         return 1

#     def get_n_out(self):
#         return 1

#     # Initialize the object
#     def init(self):
#         print("initializing object")

#     # Evaluate numerically
#     def eval(self, arg):
#         L = arg[0]  # np.array(arg[0])
#         # eigenvalues = sp.linalg.eig(L)
#         eigenvalues = np.linalg.eigvals(L)
#         sorted_eig = np.unique(eigenvalues)
#         return [sorted_eig[0]]
#         # print(eigenvalues)
#         # # print(np.unique(eigenvalues[0]))
#         # return [1]
