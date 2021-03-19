# -*- coding: utf-8 -*-
from copy import deepcopy

import networkx as nx


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


def _filter_isolated_node_candidates(indices, graph):
    """Just looking at the intermediate coordination
    environment the metal in ZIFs and prophyrins seem quite similar.
    The difference though is that when we delete the isolated metal
    in the case of ZIFs we will increase the number of connected components.
    On the other hand, in the case of prophyrins, we won't change anything.
    """
    good_node_candidates = []

    for index in indices:
        if _creates_new_connected_components(index, graph):
            good_node_candidates.append(index)

    return good_node_candidates


def _creates_new_connected_components(index, graph):
    my_graph = deepcopy(graph)
    my_graph = my_graph

    my_graph.remove_node(index)
    print(len(list(nx.connected_components(my_graph.to_undirected()))))
    return len(list(nx.connected_components(my_graph.to_undirected()))) > 1
