
from datetime import datetime, timezone
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.Platform import (
    PlatformAction,
)
import scipy.integrate as integrate


class FlockingControl:
    def __init__(
        self,
        observation: ArenaObservation,
        param_dict: dict,
        platform_dict: dict,
    ):
        self.param_dict = param_dict
        self.G_proximity = observation.graph_obs.G_communication
        self.observation = observation
        self.adjacency_mat = observation.graph_obs.adjacency_matrix_in_unit(
            unit="m", graph_type="communication"
        )  # get adjacency
        self.u_max_mps = platform_dict["u_max_in_mps"]
        self.dt_in_s = platform_dict["dt_in_s"]
        self.r_alpha = self.sigma_norm(self.param_dict["interaction_range"])
        self.d_alpha = self.sigma_norm(self.param_dict["ideal_distance"])
        self.a, self.b = [self.param_dict[key] for key in ["a", "b"]]
        self.c = np.abs(self.a - self.b) / np.sqrt(4 * self.a * self.b)
        #self.c = -self.a / np.sqrt(2 * self.a * self.b + self.b**2)
        self.vect_bump_f = np.vectorize(self.bump_function, otypes=[float])

    def sigma_norm(self, z_norm):
        val = self.param_dict["epsilon"] * (z_norm**2)
        return 1 / self.param_dict["epsilon"] * (np.sqrt(1 + val) - 1)

    def sigma_1(self, z):
        return z / np.sqrt(1 + z**2)

    def bump_function(self, z):
        # maybe can vectorize it directly with np.where(condition, val if true, val if false) but we have 3 different vals to assign for
        # 3 different conditions::
        # z = np.where(np.logical_and(z>=0, z < self.param_dict["h"]), 1, z)
        if 0 <= z < self.param_dict["h"]:
            return 1
        elif self.param_dict["h"] <= z < 1:
            arg = np.pi * (z - self.param_dict["h"]) / (1 - self.param_dict["h"])
            return 0.5 * (1 + np.cos(arg))
        else:
            return 0

    def plot_psi_and_phi_alpha(self, step: Optional[int] = 100, savefig: Optional[bool] = False):
        z_range = np.arange(start=0, stop=self.r_alpha, step=step)
        phi_alpha = [self.phi_alpha(z=z) for z in z_range]
        psi_alpha = [integrate.quad(self.phi_alpha, self.d_alpha, z)[0] for z in z_range]
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(z_range, phi_alpha)
        ax1.set_ylabel(r"$\phi_{\alpha}$")
        ax1.set_xticks([self.d_alpha, self.r_alpha])
        ax1.set_xticklabels([r"$d_{\alpha}$", r"$r_{\alpha}$"])
        ax1.grid(axis="both", linestyle="--")
        ax2.plot(z_range, psi_alpha)
        ax2.set_ylabel(r"$\psi_{\alpha}$")
        ax2.set_xticks([self.d_alpha, self.r_alpha])
        ax2.set_xticklabels([r"$d_{\alpha}$", r"$r_{\alpha}$"])
        ax2.set_xlabel(r"$\Vert z \Vert_{\sigma}$")
        ax2.grid(axis="both", linestyle="--")
        if savefig:
            plt.savefig("plot_gradient_and_potential.png")

    def phi(self, z):
        # return 0.5 * (
        #     (self.a + self.b) * self.sigma_1(z + self.c) + self.a
        # )  
        return 0.5 * ((self.a + self.b) * self.sigma_1(z + self.c) + (self.a - self.b))

    def phi_alpha(self, z):
        bump_h = self.vect_bump_f(z / self.r_alpha)
        phi = self.phi(z - self.d_alpha)
        return bump_h * phi

    def get_n_ij(self, i_node, j_neighbors, norm_q_ij):
        q_ij_lon = (
            self.observation.platform_state.lon.m[j_neighbors]
            - self.observation.platform_state.lon.m[i_node]
        )
        q_ij_lat = (
            self.observation.platform_state.lat.m[j_neighbors]
            - self.observation.platform_state.lat.m[i_node]
        )
        q_ij = np.vstack((q_ij_lon, q_ij_lat))
        return q_ij / np.sqrt(
            1 + self.param_dict["epsilon"] * (norm_q_ij.T) ** 2
        )  # columns are per neighbor
        # TODO check dimension mismatch: get a 3D array not expected

    def get_velocity_diff_array(self, node_i, neighbors_idx, sign_only: bool = False):
        velocities = np.array(self.observation.platform_state.velocity)
        # Enforce good dimensions with reshape (velocities u and v as rows)
        velocity_diff = velocities[:, node_i].reshape(2, 1) - velocities[:, neighbors_idx].reshape(
            2, neighbors_idx.size
        )
        if sign_only:
            return np.sign(velocity_diff)
        else:
            return velocity_diff

    def get_aij(self, q_ij_sigma):
        return self.vect_bump_f(q_ij_sigma / self.r_alpha)

    def get_u_i(self, node_i: int, hj_action: PlatformAction):
        neighbors_idx = np.argwhere(self.adjacency_mat[node_i, :] > 0).flatten()
        if not neighbors_idx.size:  # no neighbors
            return hj_action
        else:
            n_ij = self.get_n_ij(
                i_node=node_i,
                j_neighbors=neighbors_idx,
                norm_q_ij=self.adjacency_mat[node_i, neighbors_idx],
            )
            q_ij_sigma_norm = self.sigma_norm(z_norm=self.adjacency_mat[node_i, neighbors_idx])
            gradient_term = np.sum(self.phi_alpha(z=q_ij_sigma_norm) * n_ij, axis=1)
            velocity_match = np.matmul(
                self.get_aij(q_ij_sigma=q_ij_sigma_norm),
                self.get_velocity_diff_array(
                    node_i=node_i, neighbors_idx=neighbors_idx, sign_only=True
                ).T,
            )  # sum over neighbors where velocity is a vector (u,v)
            u_i = (
                0.5 * (gradient_term + velocity_match) * self.dt_in_s
            )  # integrate since we have a velocity input
            u_i_scaled = PlatformAction(
                min(np.linalg.norm(u_i, ord=2) / self.u_max_mps, 1),  # scale in % of max u, bounded
                direction=np.arctan2(u_i[1], u_i[0]),
            )

            return u_i_scaled + hj_action.scaling(self.param_dict["hj_factor"])


class FlockingControl2:
    def __init__(
        self,
        observation: ArenaObservation,
        param_dict: dict,
        platform_dict: dict,
    ):
        self.param_dict = param_dict
        self.G_proximity = observation.graph_obs.G_communication
        self.observation = observation
        self.adjacency_mat = observation.graph_obs.adjacency_matrix_in_unit(
            unit="m", graph_type="communication"
        )  # get adjacency
        self.u_max_mps = platform_dict["u_max_in_mps"]
        self.dt_in_s = platform_dict["dt_in_s"]
        self.r_max = self.param_dict["interaction_range"]

    def plot_potential_fcn(self, step: Optional[int] = 100, savefig: Optional[bool] = False):
        z_range = np.arange(start=0, stop=self.r_max, step=step)
        psi = [self.get_potential_function(norm_q_ij=z) for z in z_range]
        fig, ax = plt.subplots(1, 1)
        ax.plot(z_range, psi)
        ax.set_ylabel(r"$\psi$")
        ax.set_xticks([0, self.r_max/2, self.r_max])
        ax.set_xticklabels([r"$0$", r"$\frac{r_{max}}{2}$", r"$r_{max}$"])
        ax.grid(axis="both", linestyle="--")
        if savefig:
            plt.savefig("plot_gradient_and_potential.png")


    def get_potential_function(self, norm_q_ij):
        return self.r_max/(norm_q_ij*(1 - norm_q_ij/self.r_max))

    def get_analytical_gradient(self, i_node, j_neighbor):
        q_i = np.array([self.observation.platform_state.lon.m[i_node],
                        self.observation.platform_state.lat.m[i_node]])
        q_j = np.array([self.observation.platform_state.lon.m[j_neighbor],
                        self.observation.platform_state.lat.m[j_neighbor]])
        nominator = -self.r_max**2*(self.r_max*np.sign(q_i-q_j)-2*q_i + 2*q_j)
        denominator = (q_i-q_j)**2 * (self.r_max-np.abs(q_i-q_j))**2
        return nominator/denominator
    
    def get_pot_2(self, norm_q_ij):
        return norm_q_ij**2/(self.r_max-norm_q_ij)

    def get_velocity_diff_array(self, node_i, neighbors_idx, sign_only: bool = False):
        velocities = np.array(self.observation.platform_state.velocity)
        # Enforce good dimensions with reshape (velocities u and v as rows)
        velocity_diff = velocities[:, node_i].reshape(2, 1) - velocities[:, neighbors_idx].reshape(
            2, neighbors_idx.size
        )
        if sign_only:
            return np.sign(velocity_diff)
        else:
            return velocity_diff

    def get_u_i(self, node_i: int, hj_action: PlatformAction):
        neighbors_idx = np.argwhere(self.adjacency_mat[node_i, :] > 0).flatten()
        if not neighbors_idx.size:  # no neighbors
            return hj_action
        else:
            grad = 0
            for neighbor in neighbors_idx:
                grad += self.get_analytical_gradient(i_node=node_i, j_neighbor=neighbor)

            velocity_match = self.get_velocity_diff_array(
                    node_i=node_i, neighbors_idx=neighbors_idx, sign_only=True
                )
            u_i = (
                0.5 * (grad - np.sum(velocity_match)) * self.dt_in_s
            )  # integrate since we have a velocity input
            u_i_scaled = PlatformAction(
                min(np.linalg.norm(u_i, ord=2) / self.u_max_mps, 1),  # scale in % of max u, bounded
                direction=np.arctan2(u_i[1], u_i[0]),
            )

            return u_i_scaled + hj_action.scaling(self.param_dict["hj_factor"])


class RelaxedFlockingControl:
    def __init__(
        self,
        observation: ArenaObservation,
        param_dict: dict,
        platform_dict: dict,
    ):
        self.param_dict = param_dict
        self.G_proximity = observation.graph_obs.G_communication
        self.observation = observation
        self.adjacency_mat = observation.graph_obs.adjacency_matrix_in_unit(
            unit="m", graph_type="communication"
        )  # get adjacency
        self.u_max_mps = platform_dict["u_max_in_mps"]
        self.dt_in_s = platform_dict["dt_in_s"]
        self.r_alpha = self.sigma_norm(self.param_dict["interaction_range"])
        self.d_alpha = self.sigma_norm(self.param_dict["ideal_distance"])
        self.d_alpha_low = self.sigma_norm(self.param_dict["ideal_distance_low"])
        self.d_alpha_high = self.sigma_norm(self.param_dict["ideal_distance_high"])
        self.a, self.b = [self.param_dict[key] for key in ["a", "b"]]
        self.c = np.abs(self.a - self.b) / np.sqrt(4 * self.a * self.b)
        self.vect_bump_f = np.vectorize(self.bump_function, otypes=[float])
        self.vect_phi_alpha_fcn = np.vectorize(self.phi_alpha)

    def sigma_norm(self, z_norm):
        val = self.param_dict["epsilon"] * (z_norm**2)
        return 1 / self.param_dict["epsilon"] * (np.sqrt(1 + val) - 1)

    def sigma_1(self, z):
        return z / np.sqrt(1 + z**2)

    def bump_function(self, z):
        # maybe can vectorize it directly with np.where(condition, val if true, val if false) but we have 3 different vals to assign for
        # 3 different conditions::
        # z = np.where(np.logical_and(z>=0, z < self.param_dict["h"]), 1, z)
        if 0 <= z < self.param_dict["h"]:
            return 1
        elif self.param_dict["h"] <= z < 1:
            arg = np.pi * (z - self.param_dict["h"]) / (1 - self.param_dict["h"])
            return 0.5 * (1 + np.cos(arg))
        else:
            return 0

    def plot_psi_and_phi_alpha(self, step: Optional[int] = 10, savefig: Optional[bool] = False):
        z_range = np.arange(start=0, stop=self.r_alpha, step=step)
        phi_alpha = [self.vect_phi_alpha_fcn(z=z) for z in z_range]
        psi_alpha = [integrate.quad(self.vect_phi_alpha_fcn, self.d_alpha_low, z)[0] for z in z_range]
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(z_range, phi_alpha)
        ax1.set_ylabel(r"$\phi_{\alpha}$")
        ax1.set_xticks([self.d_alpha_low, self.d_alpha, self.d_alpha_high, self.r_alpha])
        ax1.set_xticklabels([r"$d_{\alpha}^{low}$", r"$d_{\alpha}$", r"$d_{\alpha}^{high}$", r"$r_{\alpha}$"])
        ax1.grid(axis="both", linestyle="--")
        ax2.plot(z_range, psi_alpha)
        ax2.set_ylabel(r"$\psi_{\alpha}$")
        ax2.set_xticks([self.d_alpha_low, self.d_alpha, self.d_alpha_high, self.r_alpha])
        ax2.set_xticklabels([r"$d_{\alpha}^{low}$", r"$d_{\alpha}$", r"$d_{\alpha}^{high}$", r"$r_{\alpha}$"])
        ax2.set_xlabel(r"$\Vert z \Vert_{\sigma}$")
        ax2.grid(axis="both", linestyle="--")
        if savefig:
            plt.savefig("plot_gradient_and_potential.png")

    def phi(self, z):
        return 0.5 * ((self.a + self.b) * self.sigma_1(z + self.c) + (self.a - self.b))

    def phi_alpha(self,z):
        if z < self.d_alpha_low:
            phi =  self.vect_bump_f(z / self.d_alpha_low)*self.phi(z - self.d_alpha_low)
        elif z >= self.d_alpha_high:
            phi = self.vect_bump_f(z / self.r_alpha)*self.phi(z - self.d_alpha_high)
        else:
            phi = 0
        return  phi

    def get_n_ij(self, i_node, j_neighbors, norm_q_ij):
        q_ij_lon = (
            self.observation.platform_state.lon.m[j_neighbors]
            - self.observation.platform_state.lon.m[i_node]
        )
        q_ij_lat = (
            self.observation.platform_state.lat.m[j_neighbors]
            - self.observation.platform_state.lat.m[i_node]
        )
        q_ij = np.vstack((q_ij_lon, q_ij_lat))
        return q_ij / np.sqrt(
            1 + self.param_dict["epsilon"] * (norm_q_ij.T) ** 2
        )  # columns are per neighbor

    def get_velocity_diff_array(self, node_i, neighbors_idx, sign_only: bool = False):
        velocities = np.array(self.observation.platform_state.velocity)
        # Enforce good dimensions with reshape (velocities u and v as rows)
        velocity_diff = velocities[:, node_i].reshape(2, 1) - velocities[:, neighbors_idx].reshape(
            2, neighbors_idx.size
        )
        if sign_only:
            return np.sign(velocity_diff)
        else:
            return velocity_diff

    def get_aij(self, q_ij_sigma):
        return self.vect_bump_f(q_ij_sigma / self.r_alpha)

    def get_u_i(self, node_i: int, hj_action: PlatformAction):
        neighbors_idx = np.argwhere(self.adjacency_mat[node_i, :] > 0).flatten()
        if not neighbors_idx.size:  # no neighbors
            return hj_action
        else:
            n_ij = self.get_n_ij(
                i_node=node_i,
                j_neighbors=neighbors_idx,
                norm_q_ij=self.adjacency_mat[node_i, neighbors_idx],
            )
            q_ij_sigma_norm = self.sigma_norm(z_norm=self.adjacency_mat[node_i, neighbors_idx])
            gradient_term = np.sum(self.vect_phi_alpha_fcn(z=q_ij_sigma_norm) * n_ij, axis=1)
            velocity_match = np.matmul(
                self.get_aij(q_ij_sigma=q_ij_sigma_norm),
                self.get_velocity_diff_array(
                    node_i=node_i, neighbors_idx=neighbors_idx, sign_only=True
                ).T,
            )  # sum over neighbors where velocity is a vector (u,v)
            u_i = (
                0.5 * (gradient_term + velocity_match) * self.dt_in_s
            ) # integrate since we have a velocity input
            u_i_scaled = PlatformAction(
                min(np.linalg.norm(u_i, ord=2) / self.u_max_mps, 1),  # scale in % of max u, bounded
                direction=np.arctan2(u_i[1], u_i[0]),
            )
            return u_i_scaled + hj_action.scaling(self.param_dict["hj_factor"])

