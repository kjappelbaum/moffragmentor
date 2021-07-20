# -*- coding: utf-8 -*-
import plotly.graph_objs as go


def ploty_plot_structure_graph(
    structure_graph, show_edges: bool = True, show_nodes: bool = True
):
    node_x = []
    node_y = []
    node_z = []

    edge_x = []
    edge_y = []
    edge_z = []

    atom_number = []

    coords = structure_graph.structure.frac_coords

    for i in range(len(coords)):
        c = coords[i]

        node_x.append(c[0])
        node_y.append(c[1])
        node_z.append(c[2])

        atom_number.append(structure_graph.structure[i].specie.number)

    for start, end, data in structure_graph.graph.edges(data=True):
        start_c = coords[start]
        end_c = coords[end] + data["to_jimage"]

        edge_x += [start_c[0], end_c[0], None]
        edge_y += [start_c[1], end_c[1], None]
        edge_z += [start_c[2], end_c[2], None]

    trace1 = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        hoverinfo="none",
        line=dict(color="black", width=2),
    )

    trace2 = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode="markers",
        hoverinfo="none",
        marker=dict(
            symbol="circle",
            size=6,
            color=atom_number,
            colorscale="Viridis",
            line=dict(color="rgb(50,50,50)", width=0.5),
        ),
    )

    axis = dict(
        showbackground=False,
        showline=False,
        zeroline=False,
        showgrid=False,
        showticklabels=False,
        title="",
    )

    layout = go.Layout(
        #          title="Network of coappearances of characters in Victor Hugo's novel<br> Les Miserables (3D visualization)",
        #          width=1000,
        #          height=1000,
        showlegend=False,
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis),
        ),
        margin=dict(t=100),
        hovermode="closest",
    )

    data = []

    if show_nodes:
        data.append(trace2)
    if show_edges:
        data.append(trace1)

    fig = go.Figure(data=data, layout=layout)
    return fig
