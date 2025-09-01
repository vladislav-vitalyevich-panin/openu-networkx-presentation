"""
    vladislav_utils.py - a collection of functions aimed at making it easier for me to work on the NetworkX presentation Jupyter Notebook.
    Contains functions for easy drawing of complex graphs, such as multigraphs and directed graphs with multiple edges between nodes, as well as just labelled graphs.
"""

import networkx as nx
import matplotlib as plt
import itertools as it

def draw_ug_networkx_with_edge_labels(G, attribute_name, pos=None):
    """
        Draw an undirected graph with its edge labels, using provided attribute name (e.g. "weight") and layout (via pos).
        
        If layout is not provided, sprint_layout is used instead with weight=None.
    """
    
    if pos is None:
        pos = nx.spring_layout(G, weight=None)

    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='black')
    labels = nx.get_edge_attributes(G,attribute_name)
    # Draw labels
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

def draw_dg_networkx_with_edge_labels(G, attribute_name, pos=None, multiple_edges_between_two_nodes=False, curvature_in_rads=0.3):
    """
        Draw a directed graph with its edge labels, using provided attribute name (e.g. "weight") and layout (via pos).
        Supports multiple non-parallel edges between two nodes via multiple_edges_between_two_nodes=True (like in multigraphs, but the edges are not "directed" towards the same node - that's not allowed via NetworkX docs).
        Modify edge curvature via curvature_in_rads (default 0.3)
        
        If layout is not provided, sprint_layout is used instead with weight=None.
    """
    
    if pos is None:
        pos = nx.spring_layout(G, weight=None)
        
    if multiple_edges_between_two_nodes:
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue')
        nx.draw_networkx_labels(G, pos) # these are node labels

        # Draw curved edges to distinguish directions
        nx.draw_networkx_edges(
            G, pos, arrowstyle='->', arrowsize=20,
            connectionstyle="arc3,rad="+str(curvature_in_rads), edge_color='black'
        )


        # Draw edge labels (relation attributes)
        edge_labels = nx.get_edge_attributes(G, attribute_name)
        augmented_draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels,
            label_pos=0.5, font_color='black', rotate=False, rad=curvature_in_rads
        )
    else:
        # Draw nodes and edges
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='black', arrows=True)
        labels = nx.get_edge_attributes(G, attribute_name)
        # Draw labels
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

def draw_labeled_multigraph(G, attr_name, pos=None, ax=None):
    """
        Draw an undirected multigraph (MultiGraph type) with its edge labels, using provided attribute name (e.g. "weight") and layout (via pos).
        
        If layout is not provided, sprint_layout is used instead with weight=None.
    """
    if pos is None:
        pos = nx.spring_layout(G)


    """
    Length of connectionstyle must be at least that of a maximum number of edges
    between pair of nodes. This number is maximum one-sided connections
    for directed graph and maximum total connections for undirected graph.
    """
    # Works with arc3 and angle3 connectionstyles
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
    # connectionstyle = [f"angle3,angleA={r}" for r in it.accumulate([30] * 4)]

    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax) #font_size=10?
    nx.draw_networkx_edges(
        G, pos, edge_color="black", connectionstyle=connectionstyle, ax=ax
    )

    labels = {
        tuple(edge): f"{attr_name}={attrs[attr_name]}"
        for *edge, attrs in G.edges(keys=True, data=True)
    }
    nx.draw_networkx_edge_labels(
        G,
        pos,
        labels,
        connectionstyle=connectionstyle,
        label_pos=0.3,
        font_color="blue",
        bbox={"alpha": 0},
        ax=ax,
    )
    

def augmented_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
):
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items