from dataclasses import dataclass


class Node:
    def __init__(self, name: str, lat: float, lon: float):
        self.name = name
        self.lat = lat
        self.lon = lon

    def __repr__(self):
        return f"Node({self.name}, {self.lat}, {self.lon})"


@dataclass
class Edge:
    source: Node | str
    destination: Node | str
    weight: float = 3
    bidirectional: bool = True


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

    def add_edge(self, edge: Edge) -> None:
        if not self.contains_edge(edge):
            if self.contains_node(edge.source) and self.contains_node(edge.destination):
                self.adjacency_dict[edge.source].append(edge)
                if edge.bidirectional:
                    self.adjacency_dict[edge.destination].append(edge)
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

    def get_edges(self) -> list:
        edges = []
        for key, value in self.adjacency_dict.items():
            if value not in edges:
                edges.append(value)
        flat_list = [item for sublist in edges for item in sublist]
        return flat_list
