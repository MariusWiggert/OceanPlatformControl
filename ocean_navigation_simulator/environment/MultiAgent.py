import copy
import networkx as nx
import logging
from typing import (
    AnyStr,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import numpy as np
import math
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformStateSet,
)
from ocean_navigation_simulator.data_sources.DataSource import DataSource
from ocean_navigation_simulator.utils import units
import datetime as dt
from functools import partial
from ocean_navigation_simulator.utils.plotting_utils import (
    get_lon_lat_time_interval_from_trajectory,
    idx_state,
)


class MultiAgent:
    def __init__(self, network_graph_dict: Dict, graph_edges: Optional[List] = None):
        self.logger = logging.getLogger("arena.multi_agent")
        self.logger.info(f"Multi-Agent Setting: Initial connections are {graph_edges}")
        self.network_prop = network_graph_dict
        if graph_edges:
            self.graph_edges = graph_edges
        self.unit_weight_edges = self.network_prop["unit"]
        self.G, self.nodes = None, None
        self.from_nodes, self.to_nodes = None, None
        self.vect_strftime = np.vectorize(dt.datetime.strftime)  # used in the plot functions
        self.list_all_edges_connections = None

    def set_graph(self, platform_set: PlatformStateSet):
        self.nodes = platform_set.get_nodes_list()
        complete_graph = nx.complete_graph(self.nodes)
        self.list_all_edges_connections = list(complete_graph.edges())
        self.from_nodes, self.to_nodes = self.split_edge_list(self.list_all_edges_connections)
        # get distance between platforms
        weights = platform_set.get_distance_btw_platforms(
            from_nodes=self.from_nodes, to_nodes=self.to_nodes
        )

        edges_remaining = [
            (node_from, node_to, weight)
            for node_from, node_to, weight in zip(self.from_nodes, self.to_nodes, weights)
            if self.in_unit(value=weight, unit=self.unit_weight_edges)
            <= self.network_prop["communication_thrsld"] + self.network_prop["margin"]
        ]
        # Build graph
        self.G = nx.Graph()
        self.G.add_nodes_from(self.nodes)
        self.G.add_weighted_edges_from(edges_remaining, weight="weight")
        self.logger.info(f"Initial connections are {list(self.G.edges())}")
        return copy.deepcopy(self.G)

    def update_graph(self, platform_set: PlatformStateSet):
        # get new distance between platforms
        weights = platform_set.get_distance_btw_platforms(
            from_nodes=self.from_nodes, to_nodes=self.to_nodes
        )
        edges_to_add = [
            (node_from, node_to, weight)
            for node_from, node_to, weight in zip(self.from_nodes, self.to_nodes, weights)
            if self.in_unit(value=weight, unit=self.unit_weight_edges)
            <= self.network_prop["communication_thrsld"] + self.network_prop["margin"]
        ]

        # update existing edges with new weigths for this timestep (and add new edges that are within communication range)
        self.G.add_weighted_edges_from(edges_to_add, weight="weight")
        # check if some edges need to be removed based on the new weights
        edges_to_remove = [
            (node_from, node_to)
            for node_from, node_to, weight in list(self.G.edges.data("weight"))
            if self.in_unit(value=weight, unit=self.unit_weight_edges)
            > self.network_prop["communication_thrsld"] + self.network_prop["margin"]
        ]
        if edges_to_remove:
            self.G.remove_edges_from(edges_to_remove)
        return copy.deepcopy(
            self.G
        )  # to save to arena: so that new elements do not change previous ones

    @staticmethod
    def in_unit(value: units.Distance, unit: str = "km"):
        if unit == "km":
            return value.km
        else:
            return value.deg

    @staticmethod
    def split_edge_list(graph_edges: List):
        return list(map(lambda x: x[0], graph_edges)), list(map(lambda x: x[1], graph_edges))

    @staticmethod
    def make_proxy(clr, mappable, **kwargs):
        # https://stackoverflow.com/a/48136768
        return Line2D([0, 1], [0, 1], color=clr, **kwargs)

    @staticmethod
    def get_integer_yticks(y_data):
        return range(min(y_data), math.ceil(max(y_data)) + 1)

    def plot_network_graph(
        self,
        G: nx,
        pos: dict,
        t_datetime: dt.datetime,
        collision_communication_thrslds: Optional[Tuple] = None,
        plot_ax_ticks: Optional[bool] = False,
        reset_plot: Optional[bool] = False,
    ):
        weight_edges = nx.get_edge_attributes(G, "weight")
        weight_edges_in_dist_unit = {
            key: self.in_unit(value, unit=self.unit_weight_edges)
            for key, value in weight_edges.items()
        }
        edge_labels = {key: f"{value.km:.1f}" for key, value in weight_edges.items()}
        # Gplot = nx.Graph()
        # Gplot.add_edges_from(G.edges())  # create a copy from G

        # reset plot this is needed for matplotlib.animation
        if reset_plot:
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
        # nx.draw_networkx_edges(Gplot, pos, width=1.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_color="white", ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

        if collision_communication_thrslds is None:
            collision_thrsld, communication_thrsld = (
                self.network_prop["collision_thrsld"],
                self.network_prop["communication_thrsld"],
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
        state_trajectory,
        multi_agent_graph_seq,
        collision_communication_thrslds: Optional[Tuple] = None,
        temporal_resolution: Optional[float] = None,
        plot_ax_ticks: Optional[bool] = False,
        output: Optional[AnyStr] = "traj_graph_anim.mp4",
        margin: float = 1,
        forward_time: bool = True,
        figsize: tuple = (10, 10),
        fps: Optional[int] = 10,
        normalize_positions: Optional[bool] = False,
    ):

        if temporal_resolution is not None:
            time_grid = np.arange(
                start=state_trajectory[0, 2, 0],
                stop=state_trajectory[0, 2, -1],
                step=temporal_resolution,
            )
        else:
            time_grid = state_trajectory[0, 2, :]

        def plot_network_render_fcn(frame_idx):
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

    def plot_distance_evolution_between_neighbors(
        self,
        list_of_graph: List[nx.Graph],
        dates: List[dt.datetime],
        neighbors_list_to_plot: Optional[List[Tuple]] = None,  # can pass also list(G.edges)
        stride_temporal_res: Optional[int] = 1,
        stride_xticks: Optional[int] = 1,
        figsize: Optional[Tuple[int]] = (8, 6),
        plot_threshold: Optional[bool] = True,
    ) -> matplotlib.axes.Axes:
        """
        Function to compute distance evolution between given neighboring nodes over specified time in dates

        """
        if neighbors_list_to_plot is None:
            neighbors_list_to_plot = self.list_all_edges_connections
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
                y=self.network_prop["communication_thrsld"] + self.network_prop["margin"],
                xmin=0,
                xmax=1,
                c="red",
                label="communication loss",
            )
            ax.axhline(
                y=self.network_prop["collision_thrsld"],
                xmin=0,
                xmax=1,
                c="purple",
                label="collision risk",
            )

        dates_xticks_str = self.vect_strftime(dates[::stride_xticks], "%d-%H:%M")
        ax.set_xticks(dates[::stride_xticks])
        ax.set_xticklabels(dates_xticks_str)
        ax.legend()
        ax.grid()
        ax.set_ylabel("distance in km")
        ax.set_title("Distance evolution between connected platforms over time")
        ax.set_xlabel("time")
        return ax

    def plot_isolated_vertices(
        self,
        list_of_graph: List[nx.Graph],
        dates: List[dt.datetime],
        stride_temporal_res: Optional[int] = 1,
        stride_xticks: Optional[int] = 1,
        figsize: Optional[Tuple[int]] = (8, 6),
    ) -> matplotlib.axes.Axes:

        """
        Function that plots the number of nodes that are isolated i.e. of degree zero over time. These nodes correspond to platforms that are disconnected from the rest of the flock

        """

        plt.figure(figsize=figsize)
        ax = plt.axes()
        isolated_nodes = list(
            map(
                lambda G: int(len(list(nx.isolates(G)))),
                list_of_graph[::stride_temporal_res],
            )
        )

        ax.step(dates[::stride_temporal_res], isolated_nodes, "--x")
        ax.set_yticks(self.get_integer_yticks(isolated_nodes))
        dates_xticks_str = self.vect_strftime(dates[::stride_xticks], "%d-%H:%M")
        ax.set_xticks(dates[::stride_xticks])
        ax.set_xticklabels(dates_xticks_str)
        ax.set_ylabel("Number of isolated platforms")
        ax.set_title("Disconnected platforms")
        ax.set_xlabel("time")
        return ax

    def plot_collision_nb_over_time(
        self,
        list_of_graph: List[nx.Graph],
        dates: List[dt.datetime],
        stride_temporal_res: Optional[int] = 1,
        stride_xticks: Optional[int] = 1,
        figsize: Optional[Tuple[int]] = (8, 6),
    ) -> matplotlib.axes.Axes:

        plt.figure(figsize=figsize)
        ax = plt.axes()
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

        ax.step(dates[::stride_temporal_res], collision_nb_list, "--x")
        ax.set_yticks(self.get_integer_yticks(collision_nb_list))
        dates_xticks_str = self.vect_strftime(dates[::stride_xticks], "%d-%H:%M")
        ax.set_xticks(dates[::stride_xticks])
        ax.set_xticklabels(dates_xticks_str)
        ax.set_ylabel("Number of collisions")
        ax.set_title("Collisions versus time")
        ax.set_xlabel("time")
        return ax
