import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np
import networkx as nx
import plotly.io as pio
from collections import defaultdict
import plotly.graph_objects as go
pio.renderers.default = "browser"
from typing import Any, List, Dict, Tuple, Union, Callable

# Note: ideally we'd like to show belief particles on hover
# https://community.plotly.com/t/displaying-image-on-point-hover-in-plotly/9223
# Not that easy with interactive python..


def plot_particles_in_2D(particle_belief, x_axis_idx, y_axis_idx,
                        x_axis_label=None,
                        y_axis_label=None,
                        true_state=None):
    """Plot the particles and the true state."""
    plt.figure()
    plt.scatter(particle_belief.states[:,x_axis_idx], particle_belief.states[:,y_axis_idx],
                s=particle_belief.weights*100)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    if true_state is not None:
        plt.scatter(true_state[x_axis_idx-3], true_state[y_axis_idx-3], s=100, c="r")
    plt.show()


def create_networkx_graph(tree_object):
    """Create a networkx graph from the tree object."""
    G = nx.DiGraph()
    # step 1: add nodes to the graph
    belief_id_tuples = [(belief_id, {"color": "blue", "belief_id": belief_id}) for belief_id in
                        np.arange(len(tree_object.belief_id_to_belief)).tolist()]
    G.add_nodes_from(belief_id_tuples)
    # add the action nodes (I number them after the belief nodes for unique IDs)
    action_id_tuples = [(action_id + len(tree_object.belief_id_to_belief),
                         {"color": "red",
                          "q_value": tree_object.action_id_to_q_values[action_id]}) for action_id in
                        np.arange(len(tree_object.action_id_to_action)).tolist()]
    G.add_nodes_from(action_id_tuples)

    # add edges to the graph based on the mcts_planner.tree.belief_id_to_action_id dictionary
    for belief_id, action_ids in tree_object.belief_id_to_action_id.items():
        if len(action_ids) != 0:
            for action_id in action_ids:
                G.add_edge(belief_id, action_id + len(tree_object.belief_id_to_belief),
                           action="a={}".format(tree_object.action_id_to_action[action_id]))
    # add edges to the graph action_id_to_belief_id_reward dictionary (action_id -> [(belief_id, reward)]
    for action_id, belief_id_reward_tuples in tree_object.action_id_to_belief_id_reward.items():
        for belief_id, reward in belief_id_reward_tuples:
            G.add_edge(action_id + len(tree_object.belief_id_to_belief), belief_id,
                       reward=reward)

    return G


def plot_tree(tree_object, node_size=1000, q_val_decimals=1, reward_decimals=1):
    """Plot the tree."""

    tree_graph = create_networkx_graph(tree_object)
    pos = graphviz_layout(tree_graph, prog="dot")
    node_color = list(nx.get_node_attributes(tree_graph, 'color').values())
    nx.draw(tree_graph, pos, with_labels=False, font_weight='bold', node_color=node_color, node_size=node_size)

    # plot the edge attributes "reward" and "action" as labels
    edge_labels_rewards = nx.get_edge_attributes(tree_graph, 'reward')
    edge_labels_rewards = {edge: "r={}".format(np.round(reward,reward_decimals)) for edge, reward in edge_labels_rewards.items()}
    edge_labels_actions = nx.get_edge_attributes(tree_graph, 'action')
    nx.draw_networkx_edge_labels(tree_graph, pos, edge_labels=edge_labels_rewards, font_color='black')
    nx.draw_networkx_edge_labels(tree_graph, pos, edge_labels=edge_labels_actions, font_color='green')

    # plot the node attribute "q_value" as a label
    q_value_labels = nx.get_node_attributes(tree_graph, 'q_value')
    # edit the dictionary q_value_labels (id -> q_value) so that q_value is a string
    q_value_labels = {node_id: "q={}".format(np.round(q_value,q_val_decimals)) for node_id, q_value in q_value_labels.items()}
    nx.draw_networkx_labels(tree_graph, pos, labels=q_value_labels)

    # now for belief nodes
    belief_node_labels = nx.get_node_attributes(tree_graph, 'belief_id')
    belief_node_labels = {node_id: "b={}".format(belief_id) for node_id, belief_id in belief_node_labels.items()}
    nx.draw_networkx_labels(tree_graph, pos, labels=belief_node_labels, font_color='white')
    plt.show()


def plot_tree_plotly(tree_object, node_size=5, q_val_decimals=1, reward_decimals=1):
    """Plot the tree using plotly."""

    tree_graph = create_networkx_graph(tree_object)
    pos = graphviz_layout(tree_graph, prog="dot")

    # plot the edge attributes "reward" and "action" as labels
    edge_labels_rewards = nx.get_edge_attributes(tree_graph, 'reward')
    edge_labels_rewards = {edge: "r={}".format(np.round(reward, reward_decimals)) for edge, reward in
                           edge_labels_rewards.items()}
    edge_labels_actions = nx.get_edge_attributes(tree_graph, 'action')

    # plot the node attribute "q_value" as a label
    q_value_labels = nx.get_node_attributes(tree_graph, 'q_value')
    # edit the dictionary q_value_labels (id -> q_value) so that q_value is a string
    q_value_labels = {node_id: "q={}".format(np.round(q_value, q_val_decimals)) for node_id, q_value in
                      q_value_labels.items()}
    # now for belief nodes
    belief_node_labels = nx.get_node_attributes(tree_graph, 'belief_id')
    belief_node_labels = {node_id: "b={}".format(belief_id) for node_id, belief_id in belief_node_labels.items()}

    merged_dict = {**belief_node_labels, **q_value_labels}
    edge_labels = {**edge_labels_rewards, **edge_labels_actions}

    vis = GraphVisualization(tree_graph, pos,
                             node_color=list(nx.get_node_attributes(tree_graph, 'color').values()),
                             node_text={**belief_node_labels, **q_value_labels},
                             edge_label={**edge_labels},
                             node_size=node_size, node_border_width=1, edge_width=0.5)
    fig = vis.create_figure(height=800, width=800, showlabel=False)
    fig.show()


def plot_belief_in_tree(tree_object, belief_state_id, x_axis_particle_idx, y_axis_particle_idx,
                        x_axis_label=None, y_axis_label=None, true_state=None):

    particle_belief = tree_object.belief_id_to_belief[belief_state_id]
    plot_particles_in_2D(particle_belief,
                         x_axis_idx=x_axis_particle_idx,
                         y_axis_idx=y_axis_particle_idx,
                        x_axis_label=x_axis_label,
                        y_axis_label=y_axis_label,
                         true_state=true_state)


Vertex = Any
Edge = Tuple[Vertex, Vertex]
Num = Union[int, float]


class GraphVisualization:
    def __init__(
        self,
        G: nx.Graph,
        pos: Dict[Vertex, Union[Tuple[Num, Num], Tuple[Num, Num, Num]]],
        node_text: Union[Dict[Vertex, str], Callable] = None,
        node_text_position: Union[Dict[Vertex, str], Callable, str] = None,
        node_text_font_color: Union[Dict[Vertex, str], Callable, str] = None,
        node_text_font_family: Union[Dict[Vertex, str], Callable, str] = None,
        node_text_font_size: Union[Dict[Vertex, Num], Callable, str] = None,
        node_size: Union[Dict[Vertex, Num], Callable, Num] = None,
        node_color: Union[Dict[Vertex, Union[str, Num]], Callable, Union[str, Num]] = None,
        node_border_width: Union[Dict[Vertex, Num], Callable, Num] = None,
        node_border_color: Union[Dict[Vertex, str], Callable, str] = None,
        node_opacity: Num = None,
        edge_width: Union[Dict[Edge, Num], Callable, Num] = None,
        edge_color: Union[Dict[Edge, str], Callable, str] = None,
        edge_opacity: Num = None,
        edge_label: Union[Dict[Edge, str], Callable, str] = None,
    ):
        # check dimensions
        if all(len(pos.get(v, [])) == 2 for v in G):
            self.is_3d = False
        elif all(len(pos.get(v, [])) == 3 for v in G):
            self.is_3d = True
        else:
            raise ValueError

        # default settings
        self.default_settings = dict(
            node_text=str,  # show node label
            node_text_position="middle center",
            node_text_font_color='#000000',
            node_text_font_family='Arial',
            node_text_font_size=14,
            node_size=10 if self.is_3d else 18,
            node_color='#fcfcfc',
            node_border_width=2,
            node_border_color='#333333',
            node_opacity=0.8,
            edge_width=4 if self.is_3d else 2,
            edge_color='#808080',
            edge_opacity=0.8,
            edge_label="no edge labels",
        )

        # save settings
        self.G = G
        self.pos = pos
        self.node_text = node_text
        self.node_text_position = node_text_position
        self.node_text_font_color = node_text_font_color
        self.node_text_font_family = node_text_font_family
        self.node_text_font_size = node_text_font_size
        self.node_size = node_size
        self.node_color = node_color
        self.node_border_width = node_border_width
        self.node_border_color = node_border_color
        self.node_opacity = node_opacity
        self.edge_width = edge_width
        self.edge_color = edge_color
        self.edge_opacity = edge_opacity
        self.edge_label = edge_label

    def _get_edge_traces(self) -> List[Union[go.Scatter, go.Scatter3d]]:
        # group all edges by (color, width)
        groups = defaultdict(list)

        for edge in self.G.edges():
            color = self._get_setting('edge_color', edge)
            width = self._get_setting('edge_width', edge)
            edge_label = self._get_setting('edge_label', edge)
            groups[(color, width, edge_label)] += [edge]

        # process each group
        traces = []
        for (color, width, edge_label), edges in groups.items():
            x, y, z = [], [], []
            x_text, y_text, z_text = [], [], []
            for v, u in edges:
                x += [self.pos[v][0], self.pos[u][0], None]
                y += [self.pos[v][1], self.pos[u][1], None]
                x_text += [(self.pos[v][0] + self.pos[u][0])/2]
                y_text += [(self.pos[v][1] + self.pos[u][1])/2]
                if self.is_3d:
                    z += [self.pos[v][2], self.pos[u][2], None]
                    z_text += [(self.pos[v][2] + self.pos[u][2])/2]

            params = dict(
                x=x,
                y=y,
                mode='lines',
                hoverinfo='none',
                line=dict(color=color, width=width),
                opacity=self._get_setting('edge_opacity'),
            )

            text_params = dict(
                x=x_text,
                y=y_text,
                marker_size=0.5,
                # make it invisible
                opacity=0,
                showlegend=False,
                text=[edge_label for _ in range(len(x_text))],
                hovertemplate='%{text}<extra></extra>',
            )

            traces += [go.Scatter3d(z=z, **params) if self.is_3d else go.Scatter(**params)]
            traces += [go.Scatter3d(z=z_text, **text_params) if self.is_3d else go.Scatter(**text_params)]

        return traces

    def _get_node_trace(self, showlabel, colorscale, showscale, colorbar_title, reversescale) -> Union[go.Scatter, go.Scatter3d]:
        x, y, z = [], [], []
        for v in self.G.nodes():
            x += [self.pos[v][0]]
            y += [self.pos[v][1]]
            if self.is_3d:
                z += [self.pos[v][2]]

        params = dict(
            x=x,
            y=y,
            mode='markers' + ('+text' if showlabel else ''),
            hoverinfo='text',
            marker=dict(
                showscale=showscale,
                colorscale=colorscale,
                reversescale=reversescale,
                color=self._get_setting('node_color'),
                size=self._get_setting('node_size'),
                line_width=self._get_setting('node_border_width'),
                line_color=self._get_setting('node_border_color'),
                colorbar=dict(
                    thickness=15,
                    title=colorbar_title,
                    xanchor='left',
                    titleside='right'
                ),
            ),
            text=self._get_setting('node_text'),
            textfont=dict(
                color=self._get_setting('node_text_font_color'),
                family=self._get_setting('node_text_font_family'),
                size=self._get_setting('node_text_font_size')
            ),
            textposition=self._get_setting('node_text_position'),
            opacity=self._get_setting('node_opacity'),
        )

        trace = go.Scatter3d(z=z, **params) if self.is_3d else go.Scatter(**params)
        return trace

    def _get_setting(self, setting_name, edge=None):
        default_setting = self.default_settings.get(setting_name)
        def_func = default_setting if callable(default_setting) else lambda x: default_setting
        setting = self.__dict__.get(setting_name)

        if edge is None:  # vertex-specific
            if setting is None:  # default is used
                if callable(default_setting):  # default is a function
                    return [def_func(v) for v in self.G.nodes()]
                else:  # default is a constant
                    return default_setting
            elif callable(setting):  # setting is a function
                return [setting(v) for v in self.G.nodes()]
            elif isinstance(setting, dict):  # setting is a dict
                return [setting.get(v, def_func(v)) for v in self.G.nodes()]
            else:  # setting is a constant
                return setting
        else:  # edge-specific
            if setting is None:  # default is used
                return def_func(edge)
            elif callable(setting):  # setting is a function
                return setting(edge)
            elif isinstance(setting, dict):  # setting is a dict
                return setting.get(edge, def_func(edge))
            else:  # setting is a constant
                return setting

    def create_figure(
        self,
        showlabel=True,
        colorscale='YlGnBu',
        showscale=False,
        colorbar_title='',
        reversescale=False,
        **params
    ) -> go.Figure:
        axis_settings = dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            visible=False,
            ticks='',
            showticklabels=False,
        )
        scene = dict(
            xaxis=axis_settings,
            yaxis=axis_settings,
            zaxis=axis_settings,
        )

        layout_params = dict(
            paper_bgcolor='rgba(255,255,255,255)',  # white
            plot_bgcolor='rgba(0,0,0,0)',  # transparent
            autosize=False,
            height=400,
            width=450 if showscale else 375,
            title='',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=5, l=0, r=0, t=20),
            annotations=[],
            xaxis=axis_settings,
            yaxis=axis_settings,
            scene=scene,
        )

        # override with the given parameters
        layout_params.update(params)

        # create figure
        fig = go.Figure(layout=go.Layout(**layout_params))
        fig.add_traces(self._get_edge_traces())
        fig.add_trace(self._get_node_trace(showlabel, colorscale, showscale, colorbar_title, reversescale))
        return fig