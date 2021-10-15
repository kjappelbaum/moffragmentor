# -*- coding: utf-8 -*-
"""This module contains functions that perform filtering on indices or fragments obtained from the other fragmentation modules"""
from copy import deepcopy
from typing import List, Tuple

import networkx as nx
import numpy as np
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import Structure

from ..utils import _get_metal_sublists, unwrap
from ..utils.periodic_graph import _get_number_of_leaf_nodes
from .molfromgraph import get_subgraphs_as_molecules


def point_in_mol_coords(point, points, lattice):
    new_coords = unwrap(np.append(points, [point], axis=0), lattice)
    return in_hull(new_coords[-1], new_coords[:-1])


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`b

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed

    https://stackoverflow.com/a/16898636
    """
    from scipy.spatial import Delaunay

    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


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


# def filter_nodes(
#     node_candidate_indices: List[List[int]],
#     graph: StructureGraph,
#     metal_indices: List[int],
#     terminal_indices: List[int],
# ) -> List[List[int]]:
#     metal_sublist = _get_metal_sublists(node_candidate_indices, metal_indices)

#     filtered_nodes, original_indices = _filter_isolated_node_candidates(
#         metal_sublist, graph, terminal_indices
#     )
#     filtered_nodes = [node_candidate_indices[i] for i in original_indices]
#     return filtered_nodes


# def _filter_isolated_node_candidates(
#     indices: List[List[int]], graph: StructureGraph, terminal_indices
# ) -> Tuple[List[List[int]], List[int]]:
#     """Just looking at the intermediate coordination
#     environment the metal in ZIFs and prophyrins seem quite similar.
#     The difference though is that when we delete the isolated metal
#     in the case of ZIFs we will increase the number of leaf nodes.
#     On the other hand, in the case of prophyrins, we won't change anything
#     (since the metal is part of a ring system).

#     Args:
#         indices (List[List[int]]): Indices for which the test is performed.
#             Typically, one would consider the metal indices of MOF nodes
#         graph (nx.Graph): structure graph on which the analysis is performed
#         terminal_indices (List[int])

#     Returns:
#         Tuple[List[List[int]], List[int]]: Filtered nodes,
#             indices from original list that "survived" the filtering process
#     """
#     good_node_candidates = []
#     good_indices = []

#     for i, index in enumerate(indices):
#         if len(index) == 1:
#             if _creates_new_leaf_nodes(
#                 index, graph.graph, terminal_indices
#             ) or _creates_new_connected_components(index, graph, terminal_indices):
#                 good_node_candidates.append(index)
#                 good_indices.append(i)
#         else:
#             good_indices.append(i)
#             good_node_candidates.append(index)
#     return good_node_candidates, good_indices


# def _creates_new_leaf_nodes(
#     indices: List[int], graph: nx.Graph, terminal_indices: List[int]
# ) -> bool:
#     my_graph = deepcopy(graph)
#     my_graph.remove_nodes_from(terminal_indices)
#     current_leaf_nodes = _get_number_of_leaf_nodes(my_graph)
#     my_graph.remove_nodes_from(indices)
#     new_leaf_nodes = _get_number_of_leaf_nodes(my_graph)

#     return new_leaf_nodes > current_leaf_nodes


# def _creates_new_connected_components(
#     indices: list, graph: StructureGraph, terminal_indices
# ) -> bool:
#     """This function tests if the removal of the nodes index by indices
#     creates new connected components. Simply put, we want to understand if
#     the removal of this part of the framework creates new "floating" components.

#     Args:
#         indices (list): node indices to test
#         graph (StructureGraph): graph on which the test is performed
#         terminal_indices (List[int])
#     Returns:
#         bool: True if there are more than 1 connected component after deletion
#             of the nodes indexed by indices
#     """
#     my_graph = deepcopy(graph)
#     my_graph.structure = Structure.from_sites(my_graph.structure.sites)
#     mol_before, _, _, _, _ = get_subgraphs_as_molecules(
#         graph,
#         filter_in_cell=False,
#         return_unique=False,
#         disable_boundary_crossing_check=True,
#     )
#     my_graph.remove_nodes(indices + terminal_indices)

#     mol, _, _, _, _ = get_subgraphs_as_molecules(
#         graph,
#         filter_in_cell=False,
#         return_unique=False,
#         disable_boundary_crossing_check=True,
#     )
#     return len(mol) > 1 & len(mol) > len(mol_before)
