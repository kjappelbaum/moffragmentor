# -*- coding: utf-8 -*-
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
from pymatgen.analysis.graphs import StructureGraph


def _get_supergraph_and_index_map(
    structure_graph: StructureGraph,
) -> Tuple[StructureGraph, Dict[int, List[int]]]:
    """Make a (3,3,3) supergraph from a structure graph and
    also compute the index mapping from original index to indices in the supergraph

    Args:
        structure_graph (StructureGraph): Input structure graph

    Returns:
        Tuple[StructureGraph, Dict[int, List[int]]]: supergraph, index mapping
    """
    supergraph = structure_graph * (3, 3, 3)
    index_map = _get_supergraph_index_map(len(structure_graph))

    return supergraph, index_map


def _get_supergraph_index_map(number_atoms: int) -> Dict[int, List[int]]:
    """create a map of nodes from original graph to its image

    Args:
        number_atoms (int): number of atoms in the structure

    Returns:
        Dict[int, List[int]]: Mapping of original index to indices in the supercell
    """
    mapping = defaultdict(list)
    # Looping over all possible replicas
    for i in range(27):
        # Adding the offset
        for n in range(number_atoms):
            mapping[n].append(n + i * number_atoms)
    return dict(mapping)


def _get_reverse_supergraph_index_map(number_atoms: int) -> Dict[int, List[int]]:
    """create a map of supercell image indices to original graph

    Args:
        number_atoms (int): number of atoms in the structure

    Returns:
        Dict[int, List[int]]: Mapping of supercell image indices to original graph
    """
    mapping = dict()
    # Looping over all possible replicas
    for i in range(27):
        # Adding the offset
        for n in range(number_atoms):
            mapping[n + i * number_atoms] = n
    return dict(mapping)


def _get_leaf_nodes(graph: nx.Graph) -> List[int]:
    return [node for node in graph.nodes() if graph.degree(node) == 1]


def _get_number_of_leaf_nodes(graph: nx.Graph) -> int:
    return len(_get_leaf_nodes(graph))


def _get_pmg_edge_dict_from_net(net) -> dict:
    pmg_edge_dict = {}

    number_linkers = len(net.linker_collection)

    for k, v in net.edge_dict.items():
        node_vertex_number = k + number_linkers
        for bound_linker in v:
            linker_idx, image, _ = bound_linker
            pmg_edge_dict[
                (node_vertex_number, linker_idx, (0, 0, 0), tuple(image))
            ] = None

    return pmg_edge_dict


def _get_colormap_for_net_sg(net):
    color_map = ["blue"] * len(net.linker_collection) + ["red"] * len(
        net.node_collection
    )
    return color_map


def _draw_net_structure_graph(net):
    color_map = _get_colormap_for_net_sg(net)
    return nx.draw(
        net.structure_graph.graph.to_undirected(),
        node_color=color_map,
        with_labels=True,
    )


def _get_pmg_structure_graph_for_net(net) -> StructureGraph:
    edge_dict = _get_pmg_edge_dict_from_net(net)
    sg = StructureGraph.with_edges(net._get_dummy_structure(), edge_dict)
    return sg
