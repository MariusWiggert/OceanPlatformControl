from dataclasses import dataclass

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import haversine_distances


class Node:
    def __init__(self, name: str, lat: float, lon: float):
        self.name = name
        self.lat = lat
        self.lon = lon

    def __repr__(self):
        return f"Node({self.name}, {self.lat}, {self.lon})"


@dataclass
class Edge:
    source: Node
    destination: Node
    weight: float = 3
    bidirectional: bool = True

    def reverse(self):
        edge = Edge(self.destination, self.source, self.weight, self.bidirectional)
        return edge


class Graph:
    def __init__(self):
        # TODO: smarter solution with only one dict?
        self.adjacency_dict = {}
        self.nodes_dict = {}

    def add_node(self, node: Node) -> Node:
        if not self.contains_node(node):
            self.adjacency_dict[node.name] = []
            self.nodes_dict[node.name] = node

    def contains_node(self, node: Node) -> bool:
        return node in self.adjacency_dict

    def remove_node(self, node: Node) -> None:
        if self.contains_node(node):
            edges = self.adjacency_dict[node]
            for edge in edges:
                self.remove_edge(edge)
            del self.adjacency_dict[node]
            del self.nodes_dict[node]

    def remove_edge(self, edge: Edge, both_directions: bool = False) -> None:
        if self.contains_edge(edge):
            self.adjacency_dict[edge.source].remove(edge)
        if both_directions and self.contains_edge(edge.reverse()):
            self.adjacency_dict[edge.destination].remove(edge.reverse())

    def add_edge(self, edge: Edge) -> None:
        if not self.contains_edge(edge):
            if self.contains_node(edge.source) and self.contains_node(edge.destination):
                self.adjacency_dict[edge.source].append(edge)
                if edge.bidirectional:
                    self.adjacency_dict[edge.destination].append(edge.reverse())
            else:
                # TODO: error handling for edges that have nodes that are not defined.
                # Problem because edges can also be created with only references (name: str)
                # pointing to node in nodes_dict.
                raise Exception("Nodes need to exist!")

    def contains_edge(self, edge: Edge) -> None:
        if self.contains_node(edge.source) and self.contains_node(edge.destination):
            source_edges = self.adjacency_dict[edge.source]
            destination_edges = self.adjacency_dict[edge.destination]
            if not edge.bidirectional:
                return edge in source_edges
            return (edge in source_edges) and (edge in destination_edges)
        else:
            return False

    def get_edges(self, both_directions: bool = True) -> list:
        edges = []
        for key, value in self.adjacency_dict.items():
            for e in value:
                if both_directions:
                    if e not in edges:
                        edges.append(e)
                else:
                    # To have bidirectional edges only once in list
                    if e not in edges and e.reverse() not in edges:
                        edges.append(e)
        return edges


# TODO: remove and use the dataclass version
def set_up_geographic_ax() -> plt.axes:
    """Helper function to set up a geographic ax object to plot on."""

    ax = plt.axes(projection=ccrs.PlateCarree())
    grid_lines = ax.gridlines(draw_labels=True, zorder=5)
    grid_lines.top_labels = False
    grid_lines.right_labels = False
    ax.add_feature(cfeature.LAND, zorder=-1000, edgecolor="black")
    return ax


def plot_graph(graph, ax, both_directions=False):
    for e in graph.get_edges(both_directions=both_directions):
        ax.plot(
            [graph.nodes_dict[e.destination].lon, graph.nodes_dict[e.source].lon],
            [graph.nodes_dict[e.destination].lat, graph.nodes_dict[e.source].lat],
            c="r",
            lw=e.weight,
            transform=ccrs.Geodetic(),  # ccrs.PlateCarree()
        )
    plt.show()


if __name__ == "__main__":
    ports = [
        ["Vancouver", 49.290, -123.11],
        ["Oakland", 37.804, -122.27],
        ["Los Angeles", 33.740, -118.28],
        ["Lazaro Cardenas", 17.927, -102.269],
        ["Honolulu", 21.309, -157.87],
        ["Tokyo", 35.619, 139.796],
        ["Shanghai", 31.220, 121.487],
        ["Singapore", 1.274, 103.802],
    ]
    lanes = [
        ["Oakland", "Los Angeles"],
        ["Oakland", "Vancouver"],
        ["Los Angeles", "Honolulu", 2],
        ["Oakland", "Honolulu", 1],
        ["Tokyo", "Honolulu"],
        ["Los Angeles", "Singapore"],
        ["Oakland", "Shanghai"],
        ["Vancouver", "Shanghai"],
        ["Lazaro Cardenas", "Shanghai"],
    ]
    graph = Graph()
    for port in ports:
        graph.add_node(Node(*port))
    for lane in lanes:
        graph.add_edge(Edge(*lane))

    # Test
    for e in graph.get_edges():
        print(
            [graph.nodes_dict[e.source].name],
            [graph.nodes_dict[e.source].lat, graph.nodes_dict[e.source].lon],
            [graph.nodes_dict[e.destination].name],
            [graph.nodes_dict[e.destination].lat, graph.nodes_dict[e.destination].lon],
        )

    # Remove all the duplicate edges (bidirectional ones from other side)
    # To avoid plotting the edges twice
    print(50 * "#")
    for e in graph.get_edges(both_directions=False):
        print(
            [graph.nodes_dict[e.source].name],
            [graph.nodes_dict[e.source].lat, graph.nodes_dict[e.source].lon],
            [graph.nodes_dict[e.destination].name],
            [graph.nodes_dict[e.destination].lat, graph.nodes_dict[e.destination].lon],
        )

    # Test plotting
    ax = set_up_geographic_ax()
    ax.set_extent([-175, -100, 15, 55], ccrs.PlateCarree())
    plot_graph(graph, ax)
