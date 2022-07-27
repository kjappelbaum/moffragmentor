# -*- coding: utf-8 -*-
"""Searches, such as depth-first search (DFS) on graphs."""
from copy import deepcopy
from typing import Dict, List, Tuple

import networkx as nx

from ..utils import _flatten_list_of_sets


def _has_path_to_any_other_metal(mof, index: int, this_metal_index: int) -> bool:
    """
    Check if some neighbor at index is only connected to this_metal_index.

    Or if we have some path to some other metal.

    A valid linker will have a path to another metal
    wheras a solvent molecule will only have a bond to `this_metal_index`

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


def recursive_dfs_until_terminal(
    mof,
    start: int,
    path: List[int] = None,  # noqa: B006
    skip_list: List[int] = None,  # noqa: B006
) -> List[int]:
    """From a given starting point perform depth-first search until leaf nodes are reached.

    Args:
        mof (MOF): A MOF instance
        start (int): Starting index for the search
        path (List[int]): Starting path. Defaults to [].
        skip_list (List[int]): List of indices to skip. Defaults to [].

    Returns:
        List[int]: Path between start and the leaf node
    """
    if path is None:
        path = []
    if skip_list is None:
        skip_list = []
    if (start not in path) and (start not in skip_list):
        path.append(start)
        if mof._is_terminal(start):  # pylint:disable=protected-access
            return path

        for neighbour in mof.get_neighbor_indices(start):
            path = recursive_dfs_until_terminal(mof, neighbour, path, skip_list)

    return path


def _complete_graph(
    mof, paths: List[List[int]], branching_nodes: List[List[int]]
) -> Tuple[List[List[int]], Dict[int, List]]:
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
        Tuple[List[List[int]], Dict[int, List[int]]]: completed graphs, paths to terminal nodes
            from branching nodes
    """
    completed_edges = []
    visited = set()
    # ToDo: Why to we do this sum thousand times in this code?
    branching_nodes = sum(branching_nodes, [])

    all_directions_to_terminal = {}
    for path in paths:
        branching_node_in_path = set(path) & set(
            branching_nodes
        )  # we need to expose those directions to complete, as we should also show them on linker/capping molecule
        directions_to_complete = []

        for branching_node in branching_node_in_path:
            for neighbour in mof.get_neighbor_indices(branching_node):
                if mof._leads_to_terminal((branching_node, neighbour)):
                    directions_to_complete.extend(
                        recursive_dfs_until_terminal(mof, neighbour, [], [branching_node])
                        + [branching_node, neighbour]
                    )
                if mof._is_terminal(neighbour):
                    directions_to_complete.extend([branching_node, neighbour])
        all_directions_to_terminal[branching_node] = directions_to_complete
        subpath = []
        for vertex in path:
            # ToDo: we can at least leverage the information about the bridges
            # we have and not just brute-force the search for all vertices
            if vertex not in visited:
                # ToDo: exclude those branching nodes that have one neighbor that leads to a terminal node
                p = recursive_dfs_until_terminal(mof, vertex, [], branching_nodes)
                subpath.extend(p)
        completed_edges.append(subpath + path + directions_to_complete)
    return completed_edges, all_directions_to_terminal


def recursive_dfs_until_branch(
    mof,
    start: int,
    path: List[int] = None,  # noqa: B006
    branching_nodes: List[int] = None,  # noqa: B006
) -> List[int]:
    """From a starting point perform DFS until branch nodes are reached.

    Args:
        mof (MOF): A MOF instance
        start (int): Starting index for the search
        path (List[int], optional): Starting path. Defaults to [].
        branching_nodes (List[int], optional): List of branching nodes.
            Defaults to [].

    Returns:
        List[int]: Path between start and the leaf node
    """
    if path is None:
        path = []
    if branching_nodes is None:
        branching_nodes = []
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


def recursive_dfs_until_cn3(
    mof,
    start: int,
    path: List[int] = None,  # noqa: B006
    branching_nodes: List[int] = None,  # noqa: B006
) -> List[int]:
    """From starting point perform DFS until CN=3 nodes are reached.

    Args:
        mof (MOF): A MOF instance
        start (int): Starting index for the search
        path (List[int], optional): Starting path. Defaults to [].
        branching_nodes (List[int], optional): List of branching nodes.
            Defaults to [].

    Returns:
        List[int]: Path between start and the leaf node
    """
    if path is None:
        path = []
    if branching_nodes is None:
        branching_nodes = []
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
    """https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements"""
    graph = nx.Graph()
    for part in paths:
        graph.add_nodes_from(part)
        graph.add_edges_from(_to_edges(part))
    graph.add_edges_from(
        _connect_connected_branching_indices(mof, _flatten_list_of_sets(branch_sites))
    )
    return graph


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
