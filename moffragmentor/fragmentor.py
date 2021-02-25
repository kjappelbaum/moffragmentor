# -*- coding: utf-8 -*-
"""Some pure functions that are used to perform the node identification
Node classification techniques described in https://pubs.acs.org/doi/pdf/10.1021/acs.cgd.8b00126
"""
from collections import OrderedDict
from copy import deepcopy

import networkx as nx
from .mof import MOF
from typing import List


def has_path_to_any_other_metal(mof: MOF, index: int, this_metal_index: int) -> bool:
    """Check if some neighbor at index is only connected to this_metal_index
        or if we have some path to some other metal

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


def recursive_dfs_until_terminal(mof: MOF, start: int, path: List[int]=[]) -> List[int]:
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


def recursive_dfs_until_branch(mof: MOF, start: int, path: List[int]=[]) -> List[int]:
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
            return path

        for neighbour in mof.get_neighbor_indices(start):
            path = recursive_dfs_until_branch(mof, neighbour, path)

    return path


def find_solvent_molecule_indices(mof: MOF, index: int, starting_metal: int) -> List[int]:
    """Finds all the indices that belong to a solvent molecule

    Args:
        mof (MOF): MOF index
        index (int): Starting index of solvent molecules
        starting_metal (int): Metal index to which solvent is bound to

    Returns:
        List[int]: List of indices that belong to solvent molecule
    """
    path = recursive_dfs_until_terminal(mof, index, [starting_metal])
    _ = path.pop(0)
    return path


def classify_neighbors(mof: MOF, node_atoms: List[int]) -> OrderedDict:
    solvent_connections = set()

    solvent_indices = []
    good_connections = set()

    for metal_index in node_atoms:
        metal_neighbors = mof.get_neighbor_indices(metal_index)

        for metal_neighbor in metal_neighbors:
            if not has_path_to_any_other_metal(mof, metal_neighbor, metal_index):
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


def fragment_all_nodes(mof: MOF, filter_out_solvent: bool = True) -> OrderedDict:
    node_atoms = set(mof.metal_indices)
    connection_index = set()

    solvent_filtered = classify_neighbors(mof, node_atoms)
    if filter_out_solvent:
        good_connections = solvent_filtered["non_solvent_connections"]
    else:
        good_connections = solvent_filtered["non_solvent_connections"] | set(
            sum(solvent_filtered["solvent_indices"], [])
        )
    new_neighbors = set(good_connections) - node_atoms

    for site in new_neighbors:
        path = recursive_dfs_until_branch(mof, site, list(node_atoms))
        connection_index.add(path[-1])
        node_atoms.update(path)

    return OrderedDict(
        [
            ("node_atoms", node_atoms),
            ("solvent_connections", solvent_filtered["solvent_connections"]),
            ("solvent_indices", solvent_filtered["solvent_indices"]),
        ]
    )


def get_oxo_node_indices(mof: MOF, filter_out_solvent: bool = True) -> OrderedDict:
    solvent_filtered = classify_neighbors(mof, mof.metal_indices)
    node_atoms = set(mof.metal_indices)
    if filter_out_solvent:
        good_connections = node_atoms
    else:
        good_connections = node_atoms | set(
            sum(solvent_filtered["solvent_indices"], [])
        )

    return OrderedDict(
        [
            ("node_atoms", good_connections),
            ("solvent_connections", solvent_filtered["solvent_connections"]),
            ("solvent_indices", solvent_filtered["solvent_indices"]),
        ]
    )
