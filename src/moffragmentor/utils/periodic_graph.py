# -*- coding: utf-8 -*-
"""Methods on structure graphs"""
import warnings
from typing import List

import networkx as nx
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import Structure

from . import _not_relevant_structure_indices


def is_periodic(mof, indices):
    graph_ = mof.structure_graph.__copy__()
    graph_.structure = Structure.from_sites(graph_.structure.sites)
    to_delete = _not_relevant_structure_indices(mof.structure, indices)
    graph_.remove_nodes(to_delete)
    mols = graph_.get_subgraphs_as_molecules()
    return len(mols) > 0


def _get_leaf_nodes(graph: nx.Graph) -> List[int]:
    return [node for node in graph.nodes() if graph.degree(node) == 1]


def _get_number_of_leaf_nodes(graph: nx.Graph) -> int:
    return len(_get_leaf_nodes(graph))


def _get_pmg_edge_dict_from_net(net: "NetEmbeding") -> dict:
    pmg_edge_dict = {}

    number_linkers = len(net.linker_collection)

    for key, value in net.edge_dict.items():
        node_vertex_number = key + number_linkers
        for bound_linker in value:
            linker_idx, image, _ = bound_linker
            pmg_edge_dict[(node_vertex_number, linker_idx, (0, 0, 0), tuple(image))] = None

    return pmg_edge_dict


def _get_colormap_for_net_sg(net: "NetEmbeding"):
    color_map = ["blue"] * len(net.linker_collection) + ["red"] * len(net.node_collection)
    return color_map


def _draw_net_structure_graph(net: "NetEmbeding"):
    color_map = _get_colormap_for_net_sg(net)
    graph = net.structure_graph.graph.to_undirected()
    return nx.draw(
        graph,
        node_color=color_map,
        with_labels=True,
    )


def _get_pmg_structure_graph_for_net(net: "NetEmbeding") -> StructureGraph:
    edge_dict = _get_pmg_edge_dict_from_net(net)
    structure_graph = StructureGraph.with_edges(
        net._get_dummy_structure(), edge_dict  # pylint: disable=protected-access
    )
    return structure_graph


def _simplify_structure_graph(structure_graph: StructureGraph) -> StructureGraph:
    """Simplifies a structure graph by removing two-connected nodes.
    We will place an edge between the nodes that were connected
    by the two-connected node.

    The function does not touch the input graph (it creates a deep copy).
    Using the deep copy simplifies the implementation a lot as we can
    add the edges in a first loop where we check for two-connected nodes
    and then remove the nodes. This avoids the need for dealing with indices
    that might change when one creates a new graph.

    Args:
        structure_graph (StructureGraph): Input structure graph.
            Usually this is a "net graph". That is, a structure graph for
            a structure in which the atoms are MOF SBUs

    Returns:
        StructureGraph: simplified structure graph, where we removed the
            two-connected nodes.
    """
    graph_copy = structure_graph.__copy__()
    to_remove = []
    added_edges = set()

    # in the first iteration we just add the edge
    # and collect the nodes to delete
    for i, _ in enumerate(structure_graph.structure):
        if structure_graph.get_coordination_of_site(i) == 2:
            # if str(structure_graph.structure[i].specie) != "Si":
            indices = []
            images = []
            for neighbor in structure_graph.get_connected_sites(i):
                indices.append(neighbor.index)
                images.append(neighbor.jimage)
                try:
                    graph_copy.break_edge(i, neighbor.index, neighbor.jimage)
                except ValueError:
                    warnings.warn("Edge cannot be broken")
            sorted_images = [x for _, x in sorted(zip(indices, images))]
            edge_tuple = (tuple(sorted(indices)), tuple(sorted_images))
            # in principle, this check should not be needed ...
            if edge_tuple not in added_edges:
                added_edges.add(edge_tuple)
                graph_copy.add_edge(indices[0], indices[1], images[0], images[1])

            to_remove.append(i)
            # else:
            #     warnings.warn(
            #         "Metal cluster with low coodination number detected.\
            #         Results might be incorrect."
            #     )

    # after we added all the edges, we can remove the nodes
    graph_copy.remove_nodes(to_remove)
    return graph_copy
