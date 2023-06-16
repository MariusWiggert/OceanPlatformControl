"""
The MultiAgent contains the logic to update the graph representation of the multi-agent network,
including computing relevant metrics and some plot functions to visualize the platform network.
"""

import copy
import dataclasses
import datetime as dt
import logging
import math
from functools import partial
from typing import (
    AnyStr,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import matplotlib
import matplotlib.animation as animation
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import simpson

from ocean_navigation_simulator.data_sources.DataSource import DataSource
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformStateSet,
)
from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator.utils.plotting_utils import (
    get_lon_lat_time_interval_from_trajectory,
    idx_state,
)


@dataclasses.dataclass
class GraphObservation:

    """
    Class that contains a graph observation: a complete graph and a communication graph which only
    contains edges with weights below a certain threshold (the platform communication threshold)

    Attributes:
        G_communication: networkx graph, where edges only exist between platforms that are within communication range
        G_complete: networkx graph with edges between all nodes

    """

    G_communication: nx.Graph
    G_complete: nx.Graph

    def adjacency_matrix_in_unit(self, unit: str, graph_type: str) -> np.ndarray:
        """To obtain the adjacency matrix in the unit of choice, as the graphs are saved with the class unit type

        Args:
            unit (str): unit to compute adjacency
            graph_type (str): choose between complete graph ("complete") and communication graph ("communication"), the latest being default

        Returns:
            np.ndarray: adjacency matrix
        """
        if graph_type == "complete":
            e = [
                (n1, n2, w.m) if unit == "m" else (n1, n2, w.km)
                for n1, n2, w in list(self.G_complete.edges.data("weight"))
            ]
        else:  # return communication graph adjacency matrix by defaut
            e = [
                (n1, n2, w.m) if unit == "m" else (n1, n2, w.km)
                for n1, n2, w in list(self.G_communication.edges.data("weight"))
            ]

        G = nx.Graph()
        G.add_nodes_from(
            self.G_complete.nodes
        )  # always add all nodes so that the adjacency matrix has the right amount of rows and cols
        G.add_weighted_edges_from(e)
        return nx.to_numpy_array(G)

    def graph_laplacian_matrix_in_unit(self, unit: str) -> np.ndarray:
        e = [
            (n1, n2, w.m) if unit == "m" else (n1, n2, w.km)
            for n1, n2, w in list(self.G_complete.edges.data("weight"))
        ]
        G = nx.Graph()
        G.add_nodes_from(
            self.G_complete.nodes
        )  # always add all nodes so that the adjacency matrix has the right amount of rows and cols
        G.add_weighted_edges_from(e)
        return nx.laplacian_matrix(G).toarray()


class MultiAgent:
    """
    MultiAgent class to create and keep information about a batch of platforms
    Connectivity computation stored as (networkx) graphs
    To interact with the arena
    """

    def __init__(self, network_graph_dict: Dict, graph_edges: Optional[List] = None):
        """Constructor

        Args:
            network_graph_dict (Dict): contains parameter for the graph, such as threshold and unit of measure
            graph_edges (Optional[List], optional): if initial graph edges want to be passed, otherwise computed in set graph. Defaults to None.
        """
        self.logger = logging.getLogger("arena.multi_agent")
        self.logger.info(f"Multi-Agent Setting: Initial connections are {graph_edges}")
        self.network_prop = network_graph_dict
        if graph_edges:
            self.graph_edges = graph_edges
        self.unit_weight_edges = self.network_prop["unit"]
        self.G_communication, self.G_complete, self.list_all_edges_connections = None, None, None
        self.from_nodes, self.to_nodes = None, None
        self.vect_strftime = np.vectorize(dt.datetime.strftime)  # used in the plot functions

    def set_graph(self, platform_set: PlatformStateSet) -> GraphObservation:
        """Resets or set for the first time graph connections and weights

        Args:
            platform_set (PlatformStateSet)

        Returns:
            GraphObservation with new weights and edges between connected platforms
        """
        nodes = platform_set.get_nodes_list()
        # Build a complete graph retaining all distances between platforms
        self.G_complete = nx.complete_graph(nodes)
        self.from_nodes, self.to_nodes = self.split_edge_list(self.G_complete.edges())
        # get distance between platforms
        weights = platform_set.get_distance_btw_platforms(
            from_nodes=self.from_nodes, to_nodes=self.to_nodes
        )
        # add weights (platform distances) for the complete graph
        self.G_complete.add_weighted_edges_from(
            [
                (from_node, to_node, weight)
                for from_node, to_node, weight in zip(self.from_nodes, self.to_nodes, weights)
            ],
            weight="weight",
        )

        # Build a communication thresholded graph (edges for which distance exceed communication threshold are discarded)
        self.G_communication = nx.Graph()
        self.G_communication.add_nodes_from(nodes)
        edges_remaining = [
            (node_from, node_to, weight)
            for node_from, node_to, weight in zip(self.from_nodes, self.to_nodes, weights)
            if self.in_unit(value=weight, unit=self.unit_weight_edges)
            < self.network_prop["communication_thrsld"]
        ]
        self.G_communication.add_weighted_edges_from(edges_remaining, weight="weight")
        self.logger.info(f"Initial connections are {list(self.G_communication.edges())}")
        return GraphObservation(
            G_communication=copy.deepcopy(self.G_communication),
            G_complete=copy.deepcopy(self.G_complete),
        )

    def update_graph(self, platform_set: PlatformStateSet) -> GraphObservation:
        """update the graphs connections based on the new positions of the platform set
            edges are added in a hysteris manner, denoted by the indicator function o(i,j)[t]=1 if the edges is to be added
            or o(i,j)[t] = 0 else. See https://linkinghub.elsevier.com/retrieve/pii/S092523121300101X for more details

        Args:
            platform_set (PlatformStateSet)

        Returns:
            GraphObservation: Communication and complete graphs with updated weights and edges
        """
        # get new distance between platforms
        weights = platform_set.get_distance_btw_platforms(
            from_nodes=self.from_nodes, to_nodes=self.to_nodes
        )
        # get distances between all platforms
        all_edges_and_weights = {key: val for key, val in zip(list(self.G_complete.edges), weights)}
        # update interconnections for the complete graph (all edges & weights used)
        self.G_complete.add_weighted_edges_from(
            [(key[0], key[1], val) for key, val in all_edges_and_weights.items()], weight="weight"
        )
        # Hysteresis based adding and removing of edges
        edges_to_remove = []
        edges_to_add_or_update = []
        for node_pair, weight in all_edges_and_weights.items():
            # o(i,j)[t-] = 1
            if node_pair in list(self.G_communication.edges):
                if (
                    self.in_unit(weight, unit=self.unit_weight_edges)
                    >= self.network_prop["communication_thrsld"]
                ):
                    edges_to_remove.append(node_pair)  # o(i,j)[t] = 0
                else:  # o(i,j)[t] = 1
                    edges_to_add_or_update.append((node_pair[0], node_pair[1], weight))
            else:  # o(i,j)[t-] = 0
                if (
                    self.in_unit(weight, unit=self.unit_weight_edges)
                    < self.network_prop["communication_thrsld"]
                    - self.network_prop["epsilon_margin"]
                ):
                    edges_to_add_or_update.append(
                        (node_pair[0], node_pair[1], weight)
                    )  # o(i,j)[t] = 1

        self.G_communication.add_weighted_edges_from(edges_to_add_or_update, weight="weight")

        if (
            edges_to_remove
        ):  # filter out the edges for which the distance between nodes is above communication threshold
            self.G_communication.remove_edges_from(edges_to_remove)
        return GraphObservation(
            G_communication=copy.deepcopy(self.G_communication),
            G_complete=copy.deepcopy(self.G_complete),
        )  # to save to arena: so that new elements do not change previous ones

    def get_isolate_nodes(
        self,
        list_of_graph: List[nx.Graph],
        stride_temporal_res: Optional[int] = 1,
    ) -> list:
        """Computes the number of isolated nodes of a graph evolving over time

        Args:
            list_of_graph (List[nx.Graph]): Graphs taken at different time instances through the simulation
            stride_temporal_res (Optional[int], optional):  Defaults to 1 if we compute the number of isolated nodes for each time

        Returns:
            list: number of isolated nodes for the graphs specified in input
        """
        return list(
            map(
                lambda G: int(len(list(nx.isolates(G)))),
                list_of_graph[::stride_temporal_res],
            )
        )

    def get_collisions(
        self,
        list_of_graph: List[nx.Graph],
        stride_temporal_res: Optional[int] = 1,
    ) -> list:
        """Computes the number of collisions of a graph evolving over time

        Args:
            list_of_graph (List[nx.Graph])
            stride_temporal_res (Optional[int], optional): Defaults to 1.

        Returns:
            list: number of collisions for the graphs specified in input
        """

        collision_nb_list = []
        for G in list_of_graph[::stride_temporal_res]:
            # count the number of collisions for each graph at datetime
            collision_nb_list.append(
                sum(
                    [
                        1
                        for n1, n2, w in G.edges(data="weight")
                        if self.in_unit(w, unit=self.unit_weight_edges)
                        < self.network_prop["collision_thrsld"]
                    ]
                )
            )
        return collision_nb_list

    # ---- Utilities methods ----#
    @staticmethod
    def in_unit(value: units.Distance, unit: str = "km") -> float:
        """
        Args:
            value (units.Distance): _description_
            unit (str, optional): _description_. Defaults to "km".

        Returns:
            float: value in the desired unit
        """
        if unit == "km":
            return value.km
        else:
            return value.deg

    @staticmethod
    def split_edge_list(graph_edges: List) -> Tuple[list]:
        """get two list for each edge ij where the first list contains
        the nodes i and the second the node j

        Args:
            graph_edges (List)

        Returns:
            Tuple[list]: two list of nodes, corresponding to the edges extremities
        """
        return list(map(lambda x: x[0], graph_edges)), list(map(lambda x: x[1], graph_edges))

    @staticmethod
    def make_proxy(clr, mappable, **kwargs) -> Line2D:
        """Function to map colors to lines and add legend to line collection

        Args:
            clr (_type_): color of the line
            mappable (_type_): plot handle

        Returns:
            Line2D: _description_
        """
        # https://stackoverflow.com/a/48136768
        return Line2D([0, 1], [0, 1], color=clr, **kwargs)

    @staticmethod
    def get_integer_yticks(y_data: list) -> range:
        """For integer ticks only for plotting

        Args:
            y_data (list):

        Returns:
            range: integer ticks in the range y_data
        """
        return range(min(y_data), math.ceil(max(y_data)) + 1)

    @staticmethod
    def LOG_insert(file: str, format: logging.Formatter, text: str, level: logging):
        """Function to insert logs, such as metrics in a log file

        Args:
            file (str): name of the file to log into
            format (logging.Formatter): format of log
            text (str): text to log
            level (logging): info, error or warning
        """
        # https://stackoverflow.com/a/62824910
        infoLog = logging.FileHandler(file)
        infoLog.setFormatter(format)
        logger = logging.getLogger(file)
        logger.setLevel(level)

        if not logger.handlers:
            logger.addHandler(infoLog)
            if level == logging.INFO:
                logger.info(text)
            if level == logging.ERROR:
                logger.error(text)
            if level == logging.WARNING:
                logger.warning(text)

        infoLog.close()
        logger.removeHandler(infoLog)
        return

    def log_metrics(
        self,
        list_of_graph: List[nx.Graph],
        dates: List[np.float64],
        success_rate_reach_target: float,
        energy_efficiency_proxy: float,
        mean_min_dist_to_target: float,
        solver_times_mean_in_s: float,
        solver_times_std_in_s: float,
        logfile: str = "logmetrics.log",
        formatLog: Optional[logging.Formatter] = None,
    ) -> dict:
        """Function to log metrics of multi-agent simulation performance

        Args:
            list_of_graph (List[nx.Graph]): graphs at different times taken during the simulation
            dates (List[dt.datetime]): times at which the graphs were recorded
            success_rate_reach_target (float): number of platforms having reached target out of all platforms
            energy_efficiency_proxy(float): average of the maximum deviation from the optimal control angle for all the platforms
                                            at each simulation step
            logfile (str, optional): name of file to log to. Defaults to "logmetrics.log".
            formatLog (Optional[logging.Formatter], optional): log format. Defaults to None.

        Returns:
            dict: also returns the metrics as a dictionary
        """
        if formatLog is None:
            formatLog = logging.Formatter(
                "%(asctime)s METRICS LOG:  %(message)s", "%Y-%m-%d %H:%M:%S"
            )

        isolated_nodes = self.get_isolate_nodes(list_of_graph=list_of_graph, stride_temporal_res=1)
        collisions = self.get_collisions(list_of_graph=list_of_graph, stride_temporal_res=1)
        dates_for_integr_s = [(t - dates[0]) for t in dates]
        integrated_communication = simpson(isolated_nodes, dates_for_integr_s) / (
            dates[-1] - dates[0]
        )
        collision_metric = sum(collisions)
        beta_connectivity_list = [len(G.edges) / len(G.nodes) for G in list_of_graph]
        initial_max_graph_degree = max([deg for _, deg in list_of_graph[0].degree])
        final_max_graph_degree = max([deg for _, deg in list_of_graph[-1].degree])
        if (
            success_rate_reach_target == 1
            and integrated_communication == 0
            and collision_metric == 0
        ):
            mission_success = 1
        else:
            mission_success = 0

        self.LOG_insert(
            logfile,
            formatLog,
            f"\nIntegral metric of isolated platforms = {integrated_communication}\
            \nNumber of collisions = {collision_metric}\
            \nFraction of platforms reaching target = {success_rate_reach_target}\
            \nMean minimum distance to target among all platforms in deg = {mean_min_dist_to_target}\
            \nAverage Beta Index = {sum(beta_connectivity_list)/len(beta_connectivity_list)}\
            \nInitial maximum degree of the graph = {initial_max_graph_degree}\
            \nFinal maximum degree of the graph = {final_max_graph_degree}\
            \nMean maximum correction from optimal control, in degrees = {energy_efficiency_proxy*180/np.pi}\
            \nMean solver time in seconds = {solver_times_mean_in_s}\
            \nStd solver time in seconds = {solver_times_std_in_s}",
            logging.INFO,
        )
        self.LOG_insert(
            logfile,
            logging.Formatter("Mission Success: %(message)s"),
            True if mission_success else False,
            logging.INFO,
        )

        return {
            "Isolated_platform_metric": integrated_communication,
            "Number_of_collision": collision_metric,
            "Reaching_target": success_rate_reach_target,
            "Mean minimum distance to target among all platforms in deg": mean_min_dist_to_target,
            "Mean maximum correction from optimal control degrees": energy_efficiency_proxy
            * 180
            / np.pi,
            "Average Beta Index": sum(beta_connectivity_list) / len(beta_connectivity_list),
            "Initial maximum degree of the graph": initial_max_graph_degree,
            "Final maximum degree of the graph": final_max_graph_degree,
            "Mission_success": mission_success,
            "mean solver time in seconds": solver_times_mean_in_s,
            "std solver time in seconds": solver_times_std_in_s,
        }

    # ---- Plot functions ----#
    def plot_network_graph(
        self,
        G: nx.Graph,
        pos: dict,
        t_datetime: dt.datetime,
        collision_communication_thrslds: Optional[Tuple] = None,
        plot_ax_ticks: Optional[bool] = False,
        reset_plot: Optional[bool] = False,
    ) -> matplotlib.axes.Axes:
        """Plot the platform set network as a graph, displaying the different communication and collisions thresholds

        Args:
            G (nx.Graph): Graph to plot
            pos (dict): positions of the nodes (x,y)
            t_datetime (dt.datetime): date at which the graph was recorded in the simulation
            collision_communication_thrslds (Optional[Tuple], optional): Defaults to None.
            plot_ax_ticks (Optional[bool], optional): Defaults to False.
            reset_plot (Optional[bool], optional): Defaults to False.

        Returns:
            matplotlib.axes.Axes
        """
        weight_edges = nx.get_edge_attributes(G, "weight")
        weight_edges_in_dist_unit = {
            key: self.in_unit(value, unit=self.unit_weight_edges)
            for key, value in weight_edges.items()
        }
        # display labels in km
        edge_labels = {key: f"{value.km:.1f}" for key, value in weight_edges.items()}
        if reset_plot:  # reset plot this is needed for matplotlib.animation
            plt.clf()
        ax = plt.axes()
        ax.set_title(f"Network graph at t= {dt.datetime.strftime(t_datetime, '%Y-%m-%d %H:%M:%S')}")
        ax.legend
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color="k",
            ax=ax,
        )
        nx.draw_networkx_labels(G, pos, font_color="white", ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

        if collision_communication_thrslds is None:
            collision_thrsld, communication_thrsld = (
                self.network_prop["collision_thrsld"],
                self.network_prop["communication_thrsld"]
                - self.network_prop["epsilon_margin"]
                * 2,  # just to visualize make the communication threshold a bit smaller
            )
        else:
            collision_thrsld, communication_thrsld = collision_communication_thrslds
        edges_comm_loss_risk = {
            key: val
            for (key, val) in weight_edges_in_dist_unit.items()
            if val > communication_thrsld
        }
        edges_collision = {
            key: val for (key, val) in weight_edges_in_dist_unit.items() if val < collision_thrsld
        }
        edges_remaining = {
            key: val
            for (key, val) in weight_edges_in_dist_unit.items()
            if key not in (edges_collision | edges_comm_loss_risk)
        }
        h1 = nx.draw_networkx_edges(
            G, pos, edgelist=list(edges_comm_loss_risk), width=4, edge_color="red", style="dashed"
        )
        h2 = nx.draw_networkx_edges(
            G, pos, edgelist=list(edges_collision), width=4, edge_color="purple"
        )
        h3 = nx.draw_networkx_edges(
            G, pos, edgelist=list(edges_remaining), width=4, edge_color="green"
        )
        proxies = [
            self.make_proxy(clr, h, linestyle=ls)
            for clr, h, ls, in zip(["red", "purple", "green"], [h1, h2, h3], ["--", "-", "-"])
        ]
        ax.legend(
            proxies, ["communication loss risk", "collision risk", "within bounds"], loc="best"
        )
        if plot_ax_ticks:
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        return ax

    def animate_graph_net_trajectory(
        self,
        state_trajectory: np.ndarray,
        multi_agent_graph_seq: list,
        collision_communication_thrslds: Optional[Tuple] = None,
        temporal_resolution: Optional[float] = None,
        plot_ax_ticks: Optional[bool] = False,
        output: Optional[AnyStr] = "traj_graph_anim.mp4",
        margin: float = 1,
        forward_time: bool = True,
        figsize: tuple = (10, 10),
        fps: Optional[int] = 5,
        normalize_positions: Optional[bool] = False,
    ):
        """Function to animate the graph connections and weights evolution during a multi-agent simulation

        Args:
            state_trajectory (np.ndarray): trajectory of the platforms: dims = (pltf_id, pltf_states, time_idx)
            multi_agent_graph_seq (list): sequence of graphs corresponding to the time indices in the state trajectory
            collision_communication_thrslds (Optional[Tuple], optional): _Defaults to None. Specified later in the plot function
            temporal_resolution (Optional[float], optional): The temporal resolution of the animation in seconds, per default will be set same as the trajectory.
            plot_ax_ticks (Optional[bool], optional):Defaults to False.
            output (Optional[AnyStr], optional): filename for the ouput. Defaults to "traj_graph_anim.mp4".
            margin (float, optional): Margin as box around nodes to plot. Defaults to 1.
            forward_time (bool, optional): play animation in forward time. Defaults to True.
            figsize (tuple, optional):  Defaults to (10, 10).
            fps (Optional[int], optional): frame per seconds. Defaults to 5.
            normalize_positions (Optional[bool], optional): if true, normalize platforms position between 0 and 1. Defaults to False.
        """
        if temporal_resolution is not None:
            time_grid = np.arange(
                start=state_trajectory[0, 2, 0],
                stop=state_trajectory[0, 2, -1],
                step=temporal_resolution,
            )
        else:
            time_grid = state_trajectory[0, 2, :]

        def plot_network_render_fcn(frame_idx: int):
            """Callable within the animation function to plot graph at each timeframe

            Args:
                frame_idx (int): frame index
            """
            posix_time = time_grid[frame_idx]
            time_idx = np.searchsorted(a=state_trajectory[0, 2, :], v=posix_time)
            t_from_idx = dt.datetime.fromtimestamp(
                state_trajectory[0, 2, time_idx], tz=dt.timezone.utc
            )
            G = multi_agent_graph_seq[time_idx]
            pos = {}
            keys = list(G.nodes)

            if normalize_positions:
                x_interval, y_interval, _ = get_lon_lat_time_interval_from_trajectory(
                    state_trajectory=np.atleast_3d(state_trajectory[:, :, time_idx]), margin=margin
                )
                var_x, var_y = (x_interval[1] - x_interval[0]), (y_interval[1] - y_interval[0])

            for node_idx in range(len(keys)):
                if (
                    normalize_positions
                ):  # if we want to have normalized node positions between 0 and 1
                    pos[keys[node_idx]] = (
                        (state_trajectory[node_idx, idx_state["lon"], time_idx] - x_interval[0])
                        / var_x,
                        (state_trajectory[node_idx, idx_state["lat"], time_idx] - y_interval[0])
                        / var_y,
                    )
                else:
                    pos[keys[node_idx]] = (
                        state_trajectory[node_idx, idx_state["lon"], time_idx],
                        state_trajectory[node_idx, idx_state["lat"], time_idx],
                    )

            self.plot_network_graph(
                G,
                pos=pos,
                t_datetime=t_from_idx,
                collision_communication_thrslds=collision_communication_thrslds,
                plot_ax_ticks=plot_ax_ticks,
                reset_plot=True,  # has to be true when animation clear figure before plotting new
            )

        render_frame = partial(plot_network_render_fcn)
        # set time direction of the animation
        frames_vector = np.where(
            forward_time, np.arange(time_grid.shape[0]), np.flip(time_grid.shape[0])
        )
        fig = plt.figure(figsize=figsize)
        ani = animation.FuncAnimation(fig, func=render_frame, frames=frames_vector, repeat=False)
        DataSource.render_animation(animation_object=ani, output=output, fps=fps)

    @staticmethod
    def set_ax_description(
        ax: matplotlib.axes.Axes,
        xticks=None,
        yticks=None,
        xtick_label=None,
        ytick_label=None,
        xlabel: str = None,
        ylabel: str = None,
        title: str = None,
        plot_legend: bool = False,
    ) -> matplotlib.axes.Axes:
        """Function to generate all the plot axes details

        Args:
            ax (matplotlib.axes.Axes):
            xticks (_type_, optional):  Defaults to None.
            yticks (_type_, optional):  Defaults to None.
            xtick_label (_type_, optional): Defaults to None.
            ytick_label (_type_, optional):  Defaults to None.
            xlabel (str, optional):  Defaults to None.
            ylabel (str, optional):  Defaults to None.
            title (str, optional): Defaults to None.
            plot_legend (bool, optional):  Defaults to False.

        Returns:
            matplotlib.axes.Axes
        """
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)
        if xtick_label is not None:
            ax.set_xticklabels(xtick_label)
        if ytick_label is not None:
            ax.set_yticklabels(ytick_label)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        if plot_legend:
            ax.legend()
        return ax

    def plot_distance_evolution_between_neighbors(
        self,
        list_of_graph: List[nx.Graph],
        dates: List[dt.datetime],
        neighbors_list_to_plot: Optional[List[Tuple]] = None,  # can pass also list(G.edges)
        stride_temporal_res: Optional[int] = 1,
        stride_xticks: Optional[int] = 1,
        figsize: Optional[Tuple[int]] = (10, 8),
        plot_threshold: Optional[bool] = True,
    ) -> matplotlib.axes.Axes:
        """Function to compute distance evolution between given neighboring nodes over specified time in dates

        Args:
            list_of_graph (List[nx.Graph]): graphs at different times taken during the simulation
            dates (List[dt.datetime]): time were graphs recorded during simulation, generally corresponds to the timesteps
            neighbors_list_to_plot (Optional[List[Tuple]], optional):  if only certain pair of platforms are to be plotted
            stride_xticks (Optional[int], optional):  Defaults to 1.
            figsize (Optional[Tuple[int]], optional): Defaults to (10, 8).
            plot_threshold (Optional[bool], optional): True if threshold horizontal lines are to be displayed

        Returns:
            matplotlib.axes.Axes: _description_
        """
        if neighbors_list_to_plot is None:
            neighbors_list_to_plot = list(self.G_complete.edges)
        plt.figure(figsize=figsize)
        ax = plt.axes()
        for neighb_tup in neighbors_list_to_plot:
            node_from, node_to = neighb_tup
            dist_list = list(
                map(
                    lambda G: G[node_from][node_to]["weight"].km
                    if neighb_tup in list(G.edges)
                    else np.nan,
                    list_of_graph[::stride_temporal_res],
                )
            )
            ax.plot(dates[::stride_temporal_res], dist_list, "--x", label=f"Pair {neighb_tup}")
        if plot_threshold:
            ax.axhline(
                y=self.network_prop["communication_thrsld"],
                xmin=0,
                xmax=1,
                c="red",
                label="communication loss high threshold",
            )
            ax.axhline(
                y=self.network_prop["communication_thrsld"] - self.network_prop["epsilon_margin"],
                xmin=0,
                xmax=1,
                c="green",
                label="communication loss low threshold",
            )
            ax.axhline(
                y=self.network_prop["collision_thrsld"],
                xmin=0,
                xmax=1,
                c="purple",
                label="collision risk",
            )

        ax = self.set_ax_description(
            ax=ax,
            xticks=dates[::stride_xticks],
            xtick_label=self.vect_strftime(dates[::stride_xticks], "%d-%H:%M"),
            ylabel="Distance in km",
            xlabel="time",
            title="Distance Evolution between platforms over time",
            plot_legend=True,
        )
        ax.grid()
        return ax

    def plot_isolated_vertices(
        self,
        ax: plt.axes,
        list_of_graph: List[nx.Graph],
        dates: List[dt.datetime],
        stride_temporal_res: Optional[int] = 1,
        stride_xticks: Optional[int] = 1,
    ) -> matplotlib.axes.Axes:
        """Function that plots the number of nodes that are isolated i.e. of degree zero over time.
            These nodes correspond to platforms that are disconnected from the rest of the flock

        Args:
            ax (plt.axes): plot on top of existing axes
            list_of_graph (List[nx.Graph]): graphs at different times taken during the simulation
            dates (List[dt.datetime]): time were graphs recorded during simulation, generally corresponds to the timesteps
            stride_temporal_res (Optional[int], optional): Defaults to 1.
            stride_xticks (Optional[int], optional): Defaults to 1.

        Returns:
            matplotlib.axes.Axes
        """
        isolated_nodes = self.get_isolate_nodes(
            list_of_graph=list_of_graph, stride_temporal_res=stride_temporal_res
        )

        ax.step(dates[::stride_temporal_res], isolated_nodes, "--x")
        ax = self.set_ax_description(
            ax=ax,
            xticks=dates[::stride_xticks],
            xtick_label=self.vect_strftime(dates[::stride_xticks], "%d-%H:%M"),
            yticks=self.get_integer_yticks(isolated_nodes),
            ylabel="Number of isolated platforms",
            xlabel="time",
            title="Disconnected platforms",
            plot_legend=False,
        )
        return ax

    def plot_collision_nb_over_time(
        self,
        ax: plt.axes,
        list_of_graph: List[nx.Graph],
        dates: List[dt.datetime],
        stride_temporal_res: Optional[int] = 1,
        stride_xticks: Optional[int] = 1,
    ) -> matplotlib.axes.Axes:
        """Function to plot the number of collisions during the simulation over time

        Args:
            ax (plt.axes): plot on top of existing axes
            list_of_graph (List[nx.Graph]): graphs at different times taken during the simulation
            dates (List[dt.datetime]): time were graphs recorded during simulation, generally corresponds to the timesteps
            stride_temporal_res (Optional[int], optional): Defaults to 1.
            stride_xticks (Optional[int], optional): Defaults to 1.

        Returns:
            matplotlib.axes.Axes
        """

        collision_nb_list = self.get_collisions(
            list_of_graph=list_of_graph, stride_temporal_res=stride_temporal_res
        )
        ax.step(dates[::stride_temporal_res], collision_nb_list, "--x")

        ax = self.set_ax_description(
            ax=ax,
            xticks=dates[::stride_xticks],
            xtick_label=self.vect_strftime(dates[::stride_xticks], "%d-%H:%M"),
            yticks=self.get_integer_yticks(collision_nb_list),
            ylabel="Number of collisions",
            xlabel="time",
            title="Collisions versus time",
            plot_legend=False,
        )
        return ax

    def plot_graph_degree(
        self,
        ax: plt.axes,
        list_of_graph: List[nx.Graph],
        dates: List[dt.datetime],
        stride_temporal_res: Optional[int] = 1,
        stride_xticks: Optional[int] = 1,
    ) -> matplotlib.axes.Axes:
        """Function to plot the graph minimum and maximum degree, over simulation time

        Args:
            ax (plt.axes): plot on top of existing axes
            list_of_graph (List[nx.Graph]): graphs at different times taken during the simulation
            dates (List[dt.datetime]): time were graphs recorded during simulation, generally corresponds to the timesteps
            stride_temporal_res (Optional[int], optional): Defaults to 1.
            stride_xticks (Optional[int], optional): Defaults to 1.

        Returns:
            matplotlib.axes.Axes
        """
        max_degree_over_t = list(
            map(
                lambda G: max([deg for _, deg in G.degree]),
                list_of_graph[::stride_temporal_res],
            )
        )
        min_degree_over_t = list(
            map(
                lambda G: min([deg for _, deg in G.degree]),
                list_of_graph[::stride_temporal_res],
            )
        )
        ax.step(dates[::stride_temporal_res], max_degree_over_t, "--xr", label="max graph degree")
        ax.step(dates[::stride_temporal_res], min_degree_over_t, "--xb", label="min graph degree")

        ax = self.set_ax_description(
            ax=ax,
            xticks=dates[::stride_xticks],
            xtick_label=self.vect_strftime(dates[::stride_xticks], "%d-%H:%M"),
            yticks=self.get_integer_yticks(max_degree_over_t + min_degree_over_t),
            ylabel="Graph degree",
            xlabel="time",
            title="Platform graph min and max degree",
            plot_legend=True,
        )
        return ax

    def plot_graph_nb_connected_components(
        self,
        ax: plt.axes,
        list_of_graph: List[nx.Graph],
        dates: List[dt.datetime],
        stride_temporal_res: Optional[int] = 1,
        stride_xticks: Optional[int] = 1,
    ) -> matplotlib.axes.Axes:
        """Function to plot the graph number of connected components over simulation time

        Args:
            ax (plt.axes): plot on top of existing axes
            list_of_graph (List[nx.Graph]): graphs at different times taken during the simulation
            dates (List[dt.datetime]): time were graphs recorded during simulation, generally corresponds to the timesteps
            stride_temporal_res (Optional[int], optional): Defaults to 1.
            stride_xticks (Optional[int], optional): Defaults to 1.

        Returns:
            matplotlib.axes.Axes
        """
        nb_connected_comp = list(
            map(
                lambda G: nx.number_connected_components(G),
                list_of_graph[::stride_temporal_res],
            )
        )
        ax.step(dates[::stride_temporal_res], nb_connected_comp, "--x")
        ax = self.set_ax_description(
            ax=ax,
            xticks=dates[::stride_xticks],
            xtick_label=self.vect_strftime(dates[::stride_xticks], "%d-%H:%M"),
            yticks=self.get_integer_yticks(nb_connected_comp),
            ylabel="Connected components",
            xlabel="time",
            title="Graph number of connected components versus time",
            plot_legend=False,
        )
        return ax
