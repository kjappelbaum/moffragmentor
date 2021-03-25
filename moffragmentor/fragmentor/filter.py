# -*- coding: utf-8 -*-
"""This module contains functions that perform filtering on indices or fragments obtained from the other fragmentation modules"""
from copy import deepcopy
from typing import List, Tuple

import networkx as nx
from pymatgen import Molecule
from pymatgen.analysis.graphs import StructureGraph

from ..utils.periodic_graph import (
    _get_number_of_leaf_nodes,
    _get_supergraph_and_index_map,
)
from .utils import _get_metal_sublists


def _filter_branch_points(
    branch_indices: list, metal_indices: list, graph: nx.Graph
) -> list:
    """In a MOF structure there might be many sites with
    more than three neighbors that do not lead to a tree or
    leaf node. The relevant branching indices are those that
    are not between other ones.
    That is, we want to filter out branching indices for which the shortest
    path to a metal goes via another branching index.

    Args:
        branch_indices (list): candidate list of branching indices
        metal_indices (list): metal indices
        graph (nx.Graph): graph on which the nodes can be access using the
            items on the branch_indices and metal_indices lists

    Returns:
        list filtered branching indices
    """
    filtered_indices = []
    for branch_index in branch_indices:
        shortest_path = _shortest_path_to_metal(branch_index, metal_indices, graph)
        if not _has_branch_index_in_path(shortest_path, branch_indices):
            filtered_indices.append(branch_index)
    return filtered_indices


def _has_branch_index_in_path(path, branch_indices):
    for metal_index in branch_indices:
        if metal_index in path:
            return True
    return False


def _shortest_path_to_metal(branch_index, metal_indices, graph):
    paths = []

    for metal_index in metal_indices:
        path = nx.shortest_path(graph, source=branch_index, target=metal_index)
        paths.append(path)

    shortest_path = min(paths, key=len)

    return shortest_path[1:]


def filter_nodes(
    node_candidate_indices: List[List[int]],
    graph: StructureGraph,
    metal_indices: List[int],
) -> List[List[int]]:
    # supergraph, index_map = _get_supergraph_and_index_map(graph)
    metal_sublist = _get_metal_sublists(node_candidate_indices, metal_indices)

    filtered_nodes, original_indices = _filter_isolated_node_candidates(
        metal_sublist, graph.graph
    )

    filtered_nodes = [node_candidate_indices[i] for i in original_indices]
    return filtered_nodes


def _filter_isolated_node_candidates(
    indices: List[List[int]], graph: nx.Graph
) -> Tuple[List[List[int]], List[int]]:
    """Just looking at the intermediate coordination
    environment the metal in ZIFs and prophyrins seem quite similar.
    The difference though is that when we delete the isolated metal
    in the case of ZIFs we will increase the number of leaf nodes.
    On the other hand, in the case of prophyrins, we won't change anything
    (since the metal is part of a ring system).

    Args:
        indices (List[List[int]]): Indices for which the test is performed.
            Typically, one would consider the metal indices of MOF nodes
        graph (nx.Graph): structure graph on which the analysis is performed

    Returns:
        Tuple[List[List[int]], List[int]]: Filtered nodes,
            indices from original list that "survived" the filtering process
    """
    good_node_candidates = []
    good_indices = []

    for i, index in enumerate(indices):
        if _creates_new_leaf_nodes(index, graph):
            good_node_candidates.append(index)
            good_indices.append(i)

    return good_node_candidates, good_indices


def _creates_new_leaf_nodes(indices: List[int], graph: nx.Graph) -> bool:
    my_graph = deepcopy(graph)
    current_leaf_nodes = _get_number_of_leaf_nodes(my_graph)
    my_graph.remove_nodes_from(indices)
    new_leaf_nodes = _get_number_of_leaf_nodes(my_graph)

    return new_leaf_nodes > current_leaf_nodes


def _creates_new_connected_components(indices: list, graph: nx.Graph) -> bool:
    """This function tests if the removal of the nodes index by indices
    creates new connected components. Simply put, we want to understand if
    the removal of this part of the framework creates new "floating" components.

    Args:
        indices (list): node indices to test
        graph (nx.Graph): graph on which the test is performed

    Returns:
        bool: True if there are more than 1 connected component after deletion
            of the nodes indexed by indices
    """
    my_graph = deepcopy(graph)
    my_graph.remove_nodes_from(indices)
    return len(list(nx.connected_components(my_graph.to_undirected()))) > 1


def is_valid_linker(linker_molecule: Molecule) -> bool:
    """A valid linker has more than one atom and at least two
    connection points

    Args:
        linker_molecule (Molecule): pymatgen molecule instance

    Returns:
        bool: true if the linker is a valid linker according to
            this definition
    """
    if len(linker_molecule) < 2:
        return False

    number_connection_sites = 0

    for linker_index in range(len(linker_molecule)):
        if linker_molecule[linker_index].properties == {"binding": True}:
            number_connection_sites += 1

    return number_connection_sites >= 2


def is_valid_node(node_molecule):
    return True
