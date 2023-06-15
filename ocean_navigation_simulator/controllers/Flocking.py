"""
    The flocking control contains the implementation of a low intereference
    safe interaction controller (LISIC) to maintain connectivity, avoid collisions
    between platforms while staying close to the optimal control provided by HJ
"""

from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.Platform import PlatformAction


class FlockingControl:
    """
    Implement flocking control for the multi-agent network of
    platforms. Uses the HJ time-optimal control input as the navigation function
    The control signal of flocking consists in a gradient of a potential function
    summed to a navigation term (from HJ). Returns a platform action object
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
        )  # indicator values if outside com. range
        self.adjacency_mat = observation.graph_obs.adjacency_matrix_in_unit(
            unit=self.param_dict["unit"], graph_type="complete"
        )  # complete adjacency matrix containing all distances between platforms
        self.u_max_mps = platform_dict["u_max_in_mps"]
        self.dt_in_s = platform_dict["dt_in_s"]
        self.r = self.param_dict["interaction_range"]
        self.list_all_platforms = list(range(self.adjacency_mat.shape[0]))
        self._set_grad_clipping_value()

    def _set_grad_clipping_value(self):
        """Set a clipping value for the gradient to avoid numerical error (going to infinity)
        The gradient is set to clip when it comes near enough to singularities such as the com. range r and 0
        The clipping value is interpolated from the gradient value at r +/- grad_clip_range
        """
        self.grad_clip_abs_val = (
            1
            / self.param_dict["epsilon"]
            * np.absolute(
                -self.r
                * (self.r - 2 * self.param_dict["grad_clip_range"])
                / (
                    self.param_dict["grad_clip_range"] ** 2
                    * (self.r - self.param_dict["grad_clip_range"]) ** 2
                )
            )
        )

    def gradient(self, norm_qij: float, inside_range: bool, grad_clip: bool = True) -> float:
        """Gradient of the potential function, clipped when the magnitude approaches infinity

        Args:
            norm_qij (float): the euclidean norm of the distance between platform i and j
            inside_range (bool): if the distance between i and j is within the interaction/communication range
            grad_clip (bool, optional): Clips the gradient starting from the gradient
                                        clipping range to the vertical asymptote. Defaults to True.

        Returns:
            float: gradient of potential function (magnitude)
        """
        if inside_range:
            if grad_clip and (
                norm_qij < self.param_dict["grad_clip_range"]
                or norm_qij > self.r - self.param_dict["grad_clip_range"]
            ):
                grad = np.sign(-self.r * (self.r - 2 * norm_qij)) * self.grad_clip_abs_val
            else:
                grad = (
                    1
                    / self.param_dict["epsilon"]
                    * (
                        -self.r
                        * (self.r - 2 * norm_qij)
                        / (norm_qij**2 * (self.r - norm_qij) ** 2)
                    )
                )
        else:
            if grad_clip and (norm_qij - self.r < self.param_dict["grad_clip_range"]):
                grad = self.grad_clip_abs_val
            else:
                grad = 1 / (2 * np.sqrt(norm_qij - self.r + self.param_dict["hysteresis"]))
        return grad

    def potential_func(self, norm_qij: float, inside_range: bool) -> float:
        """Potential function responsible for the attraction repulsion behavior between platforms

        Args:
            norm_qij (float): the euclidean norm of the distance between platform i and j
            inside_range (bool): if the distance between i and j is within the interaction/communication range

        Returns:
            float: value of the potential function
        """
        if inside_range:
            return self.r / (self.param_dict["epsilon"] * norm_qij * (self.r - norm_qij))
        else:
            return np.sqrt(norm_qij - self.r + self.param_dict["hysteresis"])

    def get_n_ij(self, i_node: int, j_neighbor: int, norm_q_ij: float) -> np.ndarray:
        """Vector along the line connecting platform i to neighboring platform j

        Args:
            i_node (int): index of platform i (node)
            j_neighbor (int): index of the neighbor
        Returns:
            np.ndarray: connecting vector, with dimensions corresponding to lon,lat
        """
        q_ij_lon = (
            self.observation.platform_state.lon.km[j_neighbor]
            - self.observation.platform_state.lon.km[i_node]
        )
        q_ij_lat = (
            self.observation.platform_state.lat.km[j_neighbor]
            - self.observation.platform_state.lat.km[i_node]
        )
        q_ij = np.vstack((q_ij_lon, q_ij_lat))
        return q_ij / norm_q_ij  # normalized vector

    @staticmethod
    def softmax(array: np.ndarray) -> np.ndarray:
        """Basic implementation of a softmax
        Subtract the maximum for numerical stability and avoid large numbers

        Args:
            array (np.ndarray): array containing the different magnitudes

        Returns:
            np.ndarray: Softmax output in the same order as the original order
        """
        e_x = np.exp(array - np.max(array))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def standard_normalization(array: np.ndarray) -> np.ndarray:
        """Standard normalization

        Args:
            array (np.ndarray): array to be normalized

        Returns:
            np.ndarray: normalized array
        """
        return array / array.sum(axis=0)

    def get_u_i(self, node_i: int, hj_action: PlatformAction) -> PlatformAction:
        """Obtain the low interference safe interaction control input. Uses a flocking
            based safe interaction controller for sustaining connectivity and to avoid colllision
            amongst agents. This safety control input is blended with an optimal control input
            here given by multi-time HJ

        Args:
            node_i (int): the platform index for which the flocking control input is
                          computed
            hj_action (PlatformAction): the time-optimal reachability control

        Returns:
            PlatformAction: flocking control signal
        """
        neighbors_idx = list(self.list_all_platforms)
        neighbors_idx.remove(node_i)  # no self-loop
        grad = 0
        for neighbor in neighbors_idx:
            n_ij = self.get_n_ij(
                i_node=node_i,
                j_neighbor=neighbor,
                norm_q_ij=self.adjacency_mat[node_i, neighbor],
            )  # obtain direction vector pointing from i to j
            grad += (
                self.gradient(
                    norm_qij=self.adjacency_mat[node_i, neighbor],
                    inside_range=bool(self.binary_adjacency[node_i, neighbor]),
                )
                * n_ij.flatten()
            )
        grad_action = PlatformAction(
            np.linalg.norm(grad, ord=2), direction=np.arctan2(grad[1], grad[0])
        )
        grad_action_unit = PlatformAction(magnitude=1, direction=grad_action.direction)
        if self.param_dict["normalization"] == "softmax":
            normalization_coeff = self.softmax(
                np.array([grad_action.magnitude, hj_action.magnitude])
            )
        else:
            normalization_coeff = self.standard_normalization(
                np.array([grad_action.magnitude, hj_action.magnitude])
            )
        return grad_action_unit.scaling(normalization_coeff[0]) + hj_action.scaling(
            normalization_coeff[1]
        )

    def plot_psi_and_phi_alpha(
        self,
        max_plot_factor: Optional[float] = 1.5,
        step: Optional[int] = 0.05,
        savefig: Optional[bool] = False,
    ):
        """Plot function to display the potential function psi and the gradient function phi

        Args:
            max_plot_factor (Optional[int], optional): _description_. Defaults to 1.5.
            step (Optional[int], optional): _description_. Defaults to 0.05.
            savefig (Optional[bool], optional): _description_. Defaults to False.
        """
        z_range = np.arange(start=0, stop=self.r * max_plot_factor, step=step)
        inside_range_arr = np.where(z_range < self.r, 1, 0)
        phi = [
            self.gradient(norm_qij=z, inside_range=indicator)
            for z, indicator in zip(z_range, inside_range_arr)
        ]
        psi = [
            self.potential_func(norm_qij=z, inside_range=indicator)
            for z, indicator in zip(z_range, inside_range_arr)
        ]
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(z_range, phi)
        ax1.set_ylabel(r"$\phi$")
        ax1.set_xticks([self.r / 2, self.r])
        ax1.set_xticklabels([r"$\frac{r}{2}$", r"$r$"])
        ax1.grid(axis="both", linestyle="--")
        ax2.plot(z_range, psi)
        ax2.set_ylabel(r"$\psi$")
        ax2.set_xticks([self.r / 2, self.r])
        ax2.set_xticklabels([r"$\frac{r}{2}$", r"$r$"])
        ax2.set_xlabel(r"$\Vert z \Vert$")
        ax2.grid(axis="both", linestyle="--")
        if savefig:
            plt.savefig("plot_gradient_and_potential.png")
