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
import numpy as np
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformStateSet,
)
from ocean_navigation_simulator.utils import units


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

    def set_graph(self, platform_set: PlatformStateSet):
        self.nodes = platform_set.get_nodes_list()
        complete_graph = nx.complete_graph(self.nodes)
        self.from_nodes, self.to_nodes = self.split_edge_list(list(complete_graph.edges()))
        # get distance between platforms
        weights = platform_set.get_distance_btw_platforms(
            from_nodes=self.from_nodes, to_nodes=self.to_nodes
        )

        edges_remaining = [
            (node_from, node_to, weight)
            for node_from, node_to, weight in zip(self.from_nodes, self.to_nodes, weights)
            if self.in_unit(value=weight)
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
            if self.in_unit(value=weight)
            <= self.network_prop["communication_thrsld"] + self.network_prop["margin"]
        ]

        # update existing edges with new weigths for this timestep (and add new edges that are within communication range)
        self.G.add_weighted_edges_from(edges_to_add, weight="weight")
        # check if some edges need to be removed based on the new weights
        edges_to_remove = [
            (node_from, node_to)
            for node_from, node_to, weight in list(self.G.edges.data("weight"))
            if self.in_unit(value=weight)
            > self.network_prop["communication_thrsld"] + self.network_prop["margin"]
        ]
        if edges_to_remove:
            self.G.remove_edges_from(edges_to_remove)
        return copy.deepcopy(
            self.G
        )  # to save to arena: so that new elements do not change previous ones

    def in_unit(self, value: units.Distance):
        if self.unit_weight_edges == "km":
            return value.km
        else:
            return value.deg

    def split_edge_list(self, graph_edges: List):
        return list(map(lambda x: x[0], graph_edges)), list(map(lambda x: x[1], graph_edges))
