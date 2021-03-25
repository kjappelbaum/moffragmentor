# -*- coding: utf-8 -*-
"""Some pure functions that are used to perform the node identification
Node classification techniques described in https://pubs.acs.org/doi/pdf/10.1021/acs.cgd.8b00126
"""
from collections import OrderedDict, namedtuple
from copy import deepcopy
from typing import List, Set

import networkx as nx

from ..utils.errors import NoMetalError
from .filter import filter_nodes

__all__ = ["find_node_clusters"]

NODELOCATION_RESULT = namedtuple(
    "NODELOCATION_RESULT", ["nodes", "branching_indices", "connecting_paths"]
)


def _has_path_to_any_other_metal(mof, index: int, this_metal_index: int) -> bool:
    """Check if some neighbor at index is only connected to this_metal_index
        or if we have some path to some other metal. A valid linker will have
        a path to another metal wheras a solvent molecule will only have a bond
        to `this_metal_index`

    Args:
        mof (MOF): A MOF instance that must provide .metal_indices and ._undirected_graph
        index (int): Index of the neighbor of interest
        this_metal_index (int): Index of the metal the neighbor is bound to

    Returns:
        bool: True if there is some path to some other metal
    """
    metal_indices = deepcopy(mof.metal_indices)
    metal_indices.remove(this_metal_index)
    g = deepcopy(mof._undirected_graph)
    g.remove_node(this_metal_index)
    for metal_index in metal_indices:
        if nx.has_path(g, index, metal_index):
            return True
    return False


def recursive_dfs_until_terminal(mof, start: int, path: List[int] = []) -> List[int]:
    """From a given starting point perform depth-first search until leaf nodes are reached

    Args:
        mof (MOF): A MOF instance
        start (int): Starting index for the search
        path (List[int], optional): Starting path. Defaults to [].

    Returns:
        List[int]: Path between start and the leaf node
    """
    if start not in path:
        path.append(start)

        if mof._is_terminal(start):
            return path

        for neighbour in mof.get_neighbor_indices(start):
            path = recursive_dfs_until_terminal(mof, neighbour, path)

    return path


def recursive_dfs_until_branch(
    mof, start: int, path: List[int] = [], branching_nodes=[]
) -> List[int]:
    """From a given starting point perform depth-first search until branch nodes are reached

    Args:
        mof (MOF): A MOF instance
        start (int): Starting index for the search
        path (List[int], optional): Starting path. Defaults to [].

    Returns:
        List[int]: Path between start and the leaf node
    """

    if start not in path:
        path.append(start)

        if mof._is_branch_point(start):
            branching_nodes.append(start)
            return path, branching_nodes

        for neighbour in mof.get_neighbor_indices(start):
            path, branching_nodes = recursive_dfs_until_branch(
                mof, neighbour, path, branching_nodes
            )

    return path, branching_nodes


def find_solvent_molecule_indices(mof, index: int, starting_metal: int) -> List[int]:
    """Finds all the indices that belong to a solvent molecule

    Args:
        mof (MOF) index
        index (int): Starting index of solvent molecules
        starting_metal (int): Metal index to which solvent is bound to

    Returns:
        List[int]: List of indices that belong to solvent molecule
    """
    path = recursive_dfs_until_terminal(mof, index, [starting_metal])
    _ = path.pop(0)
    return path


def classify_neighbors(mof, node_atoms: Set[int]) -> OrderedDict:
    solvent_connections = set()

    solvent_indices = []
    good_connections = set()

    for metal_index in node_atoms:
        metal_neighbors = mof.get_neighbor_indices(metal_index)

        for metal_neighbor in metal_neighbors:
            if not _has_path_to_any_other_metal(mof, metal_neighbor, metal_index):
                solvent_connections.add(metal_neighbor)
                solvent_indices.append(
                    find_solvent_molecule_indices(mof, metal_neighbor, metal_index)
                )
            else:
                good_connections.add(metal_neighbor)

    return OrderedDict(
        [
            ("solvent_indices", solvent_indices),
            ("solvent_connections", solvent_connections),
            ("non_solvent_connections", good_connections),
        ]
    )


def _to_graph(l):
    G = nx.Graph()
    for part in l:
        G.add_nodes_from(part)
        G.add_edges_from(_to_edges(part))
    return G


def _to_edges(l):
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current


def find_node_clusters(mof) -> NODELOCATION_RESULT:
    """This function locates the branchin indices, and node clusters in MOFs.
    Starting from the metal indices it performs depth first search on the structure
    graph up to branching points.

    Args:
        mof (MOF): moffragmentor MOF instance

    Returns:
        NODELOCATION_RESULT: nametuple with the slots "nodes", "branching_indices" and
            "connecting_paths"

    Raises:
        NoMetalError: In case the structure does not contain any metal. The presence of
            a metal is crucial for the fragmentation algorithm.
    """
    paths = []
    branch_sites = []

    connecting_paths_ = set()

    # From every metal index in the structure perform DFS up to a
    # branch point
    for metal_index in mof.metal_indices:
        p, b = recursive_dfs_until_branch(mof, metal_index, [], [])
        paths.append(p)
        branch_sites.append(b)

    # Todo: maybe filter there the paths to only consider those that end at a valid branch point
    # we find the connected components in those paths
    g = _to_graph(paths)
    nodes = list(nx.connected_components(g))
    # filter out "node" candidates that are not actual nodes.
    # in practice this is relevant for ligands with metals in them (e.g., porphyrins)
    nodes = filter_nodes(nodes, mof.structure_graph, mof.metal_indices)

    bs = set(sum(branch_sites, []))
    # we store the shortest paths between nodes and branching indices
    for metal, branch_sites_for_metal in zip(mof.metal_indices, branch_sites):
        for branch_site in branch_sites_for_metal:
            connecting_paths_.update(
                sum(
                    nx.all_shortest_paths(mof._undirected_graph, metal, branch_site), []
                )
            )

    # from the connecting paths we remove the metal indices and the branching indices
    connecting_paths_ -= set(mof.metal_indices)
    connecting_paths_ -= bs

    res = NODELOCATION_RESULT(nodes, bs, connecting_paths_)
    return res
