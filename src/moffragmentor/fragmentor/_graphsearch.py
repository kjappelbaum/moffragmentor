# -*- coding: utf-8 -*-
"""Searches, such as depth-first search (DFS) on graphs."""
from copy import deepcopy
from typing import List

import networkx as nx

from ..utils import _flatten_list_of_sets


def _has_path_to_any_other_metal(mof, index: int, this_metal_index: int) -> bool:
    """Check if some neighbor at index is only connected to this_metal_index
        or if we have some path to some other metal.

        A valid linker will have a path to another metal wheras a solvent molecule will only have a bond
        to `this_metal_index`

    Args:
        mof (MOF): A MOF instance that must provide
            .metal_indices and ._undirected_graph
        index (int): Index of the neighbor of interest
        this_metal_index (int): Index of the metal the neighbor is bound to

    Returns:
        bool: True if there is some path to some other metal
    """
    metal_indices = deepcopy(mof.metal_indices)
    metal_indices.remove(this_metal_index)
    graph = deepcopy(mof.nx_graph)
    graph.remove_node(this_metal_index)
    for metal_index in metal_indices:
        if nx.has_path(graph, index, metal_index):
            return True
    return False


def recursive_dfs_until_terminal(  # pylint:disable=dangerous-default-value
    mof,
    start: int,
    path: List[int] = [],
    skip_list: List[int] = [],
) -> List[int]:
    """From a given starting point perform depth-first search
    until leaf nodes are reached

    Args:
        mof (MOF): A MOF instance
        start (int): Starting index for the search
        path (List[int], optional): Starting path. Defaults to [].

    Returns:
        List[int]: Path between start and the leaf node
    """
    if (start not in path) and (start not in skip_list):
        path.append(start)
        if mof._is_terminal(start):  # pylint:disable=protected-access
            return path

        for neighbour in mof.get_neighbor_indices(start):
            path = recursive_dfs_until_terminal(mof, neighbour, path, skip_list)

    return path


def _complete_graph(
    mof, paths: List[List[int]], branching_nodes: List[List[int]]
) -> List[List[int]]:
    """Loop over all paths that were traversed in DFS.

    Then, if the verices are not branching add all the paths
    that lead to terminal nodes. That is, in this step, we're adding
    the "capping" components of MOF nodes like capping formates or
    µ(3)-OH centers.

    Args:
        mof (MOF): Instance of a MOF object
        paths (List[List[int]]): Paths traversed in the initial DFS
            from metal indices to branching nodes
        branching_nodes (List[List[int]]): nodes that were
            identified as branching nodes

    Returns:
        List[List[int]]: completed graphs
    """
    completed_edges = []
    visited = set()
    # ToDo: Why to we do this sum thousand times in this code?
    branching_nodes = sum(branching_nodes, [])
    for path in paths:
        subpath = []
        for vertex in path:
            # ToDo: we can at least leverage the information about the bridges
            # we have and not just brute-force the search for all vertices
            if vertex not in visited:
                p = recursive_dfs_until_terminal(mof, vertex, [], branching_nodes)
                subpath.extend(p)
        completed_edges.append(subpath + path)
    return completed_edges


def recursive_dfs_until_branch(  # pylint:disable=dangerous-default-value
    mof,
    start: int,
    path: List[int] = [],
    branching_nodes=[],
) -> List[int]:
    """From a given starting point perform depth-first search
    until branch nodes are reached

    Args:
        mof (MOF): A MOF instance
        start (int): Starting index for the search
        path (List[int], optional): Starting path. Defaults to [].

    Returns:
        List[int]: Path between start and the leaf node
    """

    if start not in path:
        path.append(start)

        if mof._is_branch_point(start):  # pylint:disable=protected-access
            branching_nodes.append(start)
            return path, branching_nodes

        for neighbour in mof.get_neighbor_indices(start):
            path, branching_nodes = recursive_dfs_until_branch(
                mof, neighbour, path, branching_nodes
            )

    return path, branching_nodes


def recursive_dfs_until_cn3(  # pylint:disable=dangerous-default-value
    mof,
    start: int,
    path: List[int] = [],
    branching_nodes=[],
) -> List[int]:
    """From a given starting point perform depth-first search
    until CN=3 nodes are reaches

    Args:
        mof (MOF): A MOF instance
        start (int): Starting index for the search
        path (List[int], optional): Starting path. Defaults to [].

    Returns:
        List[int]: Path between start and the leaf node
    """

    if start not in path:
        path.append(start)

        if (len(mof.get_neighbor_indices(start)) >= 3) & (
            start not in mof.metal_indices
        ):  # pylint:disable=protected-access
            branching_nodes.append(start)
            return path, branching_nodes

        for neighbour in mof.get_neighbor_indices(start):
            path, branching_nodes = recursive_dfs_until_cn3(mof, neighbour, path, branching_nodes)

    return path, branching_nodes


def _to_graph(mof, paths, branch_sites):
    """https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements"""  # pylint:disable=line-too-long
    G = nx.Graph()
    for part in paths:
        G.add_nodes_from(part)
        G.add_edges_from(_to_edges(part))
    G.add_edges_from(_connect_connected_branching_indices(mof, _flatten_list_of_sets(branch_sites)))
    return G


def _to_edges(paths):
    iterator = iter(paths)
    last = next(iterator)  # pylint:disable=stop-iteration-return

    for current in iterator:
        yield last, current
        last = current


def _connect_connected_branching_indices(mof, flattend_path) -> set:
    edges = set()

    for i in flattend_path:
        neighbors = mof.get_neighbor_indices(i)
        for neighbor in neighbors:
            if neighbor in flattend_path:
                edges.add(tuple(sorted((i, neighbor))))

    return edges
