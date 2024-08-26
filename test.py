from pnpxai.core.detector import symbolic_trace, extract_graph_data
from tutorials.helpers import get_torchvision_model
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import plotly.graph_objects as go


# model, transform = get_torchvision_model("vit_b_16")
model, transform = get_torchvision_model("resnet18")
graph_data = extract_graph_data(symbolic_trace(model))

G = nx.DiGraph()
for node in graph_data['nodes']:
    G.add_node(node['name'])

all_roots = [node for node in G if G.in_degree(node) == 0]

for edge in graph_data['edges']:
    if edge['source'] in G.nodes and edge['target'] in G.nodes:
        G.add_edge(edge['source'], edge['target'])


def get_pos1(graph):
    graph = graph.copy()
    for layer, nodes in enumerate(reversed(tuple(nx.topological_generations(graph)))):
        for node in nodes:
            graph.nodes[node]["layer"] = layer

    pos = nx.multipartite_layout(graph, subset_key="layer", align='horizontal')
    return pos


def set_depths(graph: nx.DiGraph, node: str, node_levels: dict, depth: int = 0):
    # Compute the number of nodes for each level
    if node in node_levels:
        return node_levels

    node_levels[node] = depth
    for child in graph.successors(node):
        node_levels = set_depths(graph, child, node_levels, depth + 1)

    return node_levels


def set_cols(graph: nx.DiGraph, node: str, node_cols: dict, left: float = 0., right: float = 1.):
    if node in node_cols:
        node_cols[node] = right
        return node_cols

    node_cols[node] = right
    children = list(graph.successors(node))
    if len(children) == 0:
        return node_cols

    width = (right - left) / len(children)
    for i, child in enumerate(children):
        node_cols = set_cols(graph, child, node_cols,
                             left + i * width, left + (i + 1) * width)

    return node_cols


def get_pos2(graph: nx.DiGraph, width: float = 1., height: float = 1.):
    '''
       G: the graph
       root: the root node
       width: horizontal space allocated for drawing
       height: vertical space allocated for drawing
    '''

    # Extract all graph roots
    graph_roots = [node for node in graph if graph.in_degree(node) == 0]

    # Compute the depth of each node
    node_depths = {}
    for node in graph_roots:
        node_depths = set_depths(graph, node, node_depths)

    # Compute hierarchical horizontal positions
    node_cols = {}
    init_width = 1 / len(graph_roots)
    for i, node in enumerate(graph_roots):
        node_cols = set_cols(
            graph, node, node_cols, left=i * init_width, right=(i + 1) * init_width
        )

    max_depth = max(node_depths.values())
    vert_gap = height / max_depth
    pos = {}

    depth_nodes = [set() for _ in range(max_depth + 1)]
    for node in graph.nodes:
        depth = node_depths[node]
        depth_nodes[depth].add(node)

    for depth, nodes in enumerate(depth_nodes):
        if len(nodes) == 0:
            continue

        for node in nodes:
            pos[node] = (
                node_cols[node] * width,
                - vert_gap * depth
            )

    return pos


def plot_pyplot(graph, pos):
    fig = plt.figure(figsize=(12, 24))
    ax = fig.gca()
    nx.draw(graph, pos=pos, with_labels=True, node_size=60, font_size=8, ax=ax)
    return fig


def plot_plotly(graph: nx.DiGraph, pos: dict):
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode="markers+lines",
        marker=dict(
            symbol="arrow",
            color="#888",
            size=8,
            angleref="previous",
            standoff=4,
        ),
    )

    node_x = []
    node_y = []
    node_text = []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
        title='Model Architecture',
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    )
    return fig


pos = get_pos1(G)
plot_pyplot(G, pos, 'hierarchy1.png')
pos = get_pos2(G)
plot_plotly(G, pos, 'hierarchy2.png')
