"""
Routines for finding branching points in a structure graph of a MOF.

Note that those routines do not work for other reticular materials
as they assume the presence of a metal.
"""
from operator import itemgetter
from typing import List, Tuple

import networkx as nx
import numpy as np
from loguru import logger
from more_itertools import pairwise
from pymatgen.core import Structure
from structuregraph_helpers.subgraph import get_subgraphs_as_molecules

from moffragmentor.utils import _not_relevant_structure_indices

from .. import mof

__all__ = ("get_branch_points",)


def get_distances_to_metal(
    mof: "mof.MOF", site: int
) -> List[float]:  # noqa: F821 - forward reference
    """For a given site, return the distances to all metals in the MOF."""
    distances = []
    for i in mof.metal_indices:
        distances.append(mof.structure.get_distance(site, i))
    return distances


def has_bridge_in_path(mof: "mof.MOF", path: List[int]) -> bool:
    return any(mof._leads_to_terminal(edge) for edge in pairwise(path))


def get_two_edge_paths_from_site(mof: "mof.MOF", site: int) -> List[List[int]]:
    """
    Return all two edge paths from a site.

    Args:
        mof (MOF): MOF object
        site (int): index of site that is to be probed.

    Returns:
        List[List[int]]: List of all two edge paths from the site.

    Example:
        >>> mof = MOF(...)
        >>> get_two_edge_paths_from_site(mof, 0)
        [[0, 1, 2], [0, 3, 4], [0, 5, 6]]
    """
    paths = []
    for i in mof.get_neighbor_indices(site):
        for j in mof.get_neighbor_indices(i):
            if j != site:
                paths.append([site, i, j])

    return paths


def has_metal_in_path(mof: "MOF", path: List[int]) -> bool:  # noqa: F821 - forward reference
    """Return True if the path contains a metal."""
    return any(site in mof.metal_indices for site in path)


def has_non_bridge_path_with_metal(mof: "mof.MOF", site: int) -> bool:
    """Return True if the MOF has a non-bridge path with a meta.l"""
    return any(
        (not has_bridge_in_path(mof, path)) and (has_metal_in_path(mof, path))
        for path in get_two_edge_paths_from_site(mof, site)
    )


def _is_branch_point(mof: "mof.MOF", index: int, allow_metal: bool = False) -> bool:
    """Check if a site is a branching point.

    The branch point definition is key for splitting MOFs into linker and nodes.
    Branch points are here defined as points
    that have at least three connections that do not lead to a tree or
    leaf node.

    Args:
        mof: MOF object
        index (int): index of site that is to be probed
        allow_metal (bool): If True it does not perform
            this check for metals (and just return False). Defaults to False.

    Returns:
        bool: True if this is a branching index
    """
    connected_sites = mof.get_neighbor_indices(index)

    if len(connected_sites) < 3:
        return False

    if (not allow_metal) and (index in mof.metal_indices):
        return False

    # lets store all the info in a numpy array
    sites = []
    for connected_site in connected_sites:
        leads_to_terminal = mof._leads_to_terminal((index, connected_site))
        is_terminal = mof._is_terminal(connected_site)
        in_metal_indices = connected_site in mof.metal_indices
        non_bridge_metal = has_non_bridge_path_with_metal(mof, index)
        sites.append([leads_to_terminal, is_terminal, in_metal_indices, non_bridge_metal])
    sites = np.array(sites).astype(bool)
    non_terminal_metal_connections = np.sum(~sites[:, 0] & ~sites[:, 1] & sites[:, 3])
    non_terminal_non_metal_connections = np.sum(~sites[:, 0] & ~sites[:, 1] & ~sites[:, 2])
    terminal_metal_connections = np.sum((sites[:, 0] | sites[:, 1]) & sites[:, 2])

    if terminal_metal_connections > 0:
        return False
    if (non_terminal_non_metal_connections >= 2) & (non_terminal_metal_connections > 0):
        return True

    return False


def filter_branch_points(
    mof: "mof.MOF", branching_indices  # noqa: F821 - forward reference
) -> List[int]:
    """Return a list of all branching points in the MOF."""
    # now, check if there are connected branching points
    # we need to clean that up as it does not make sense for splitting and the clustering
    # algorithm will not work (i believe).
    graph_ = mof.structure_graph.__copy__()
    graph_.structure = Structure.from_sites(graph_.structure.sites)
    to_delete = _not_relevant_structure_indices(mof.structure, branching_indices)
    graph_.remove_nodes(to_delete)
    mols, graphs, idx, centers, coordinates = get_subgraphs_as_molecules(
        graph_, return_unique=False
    )

    verified_indices = []

    for _, graph, index in zip(mols, graphs, idx):
        if len(index) == 1:
            verified_indices.extend(index)
        else:
            (
                good_by_distance,
                not_resolvable_by_distance,
                not_resolvable_graphs,
            ) = remove_not_closest_neighbor(mof, graph.graph, index)
            verified_indices.extend(good_by_distance)
            for not_resolvable_graph, not_resolvable_by_distance_index in zip(
                not_resolvable_graphs, not_resolvable_by_distance
            ):
                verified_indices.extend(
                    cluster_nodes(not_resolvable_graph, not_resolvable_by_distance_index, mof)
                )

    return verified_indices


def remove_not_closest_neighbor(
    mof: "mof.MOF", graph: nx.Graph, index: int  # noqa: F821 - forward reference
) -> List[int]:
    """Remove all nodes that are not the closest neighbor to the given index."""
    good_indices = []
    not_resolvable = []
    not_resolvable_graphs = []
    # divide indices into two groups, one with the closest neighbor metal and one with the rest
    minimum_distance, not_minimum_distance = rank_by_metal_distance(index, mof)
    # delete all nodes that are not the closest neighbor

    graph.remove_nodes_from(not_minimum_distance)
    # now, get new connected components
    for g in nx.connected_components(graph.to_undirected()):
        if len(g) == 1:
            good_indices.append(index[list(g)[0]])
        else:
            not_resolvable.append(itemgetter(*list(g))(index))
            not_resolvable_graphs.append(graph.subgraph(g))
    if len(good_indices) == 0:
        logger.warning("Could not find unique branching point")
    return good_indices, not_resolvable, not_resolvable_graphs


def has_bond_to_metal(mof: "MOF", index: int) -> bool:  # noqa: F821 - forward reference
    """Return True if the site has a bond to a metal."""
    return any(i in mof.metal_indices for i in mof.get_neighbor_indices(index))


def rank_by_metal_distance(idx, mof) -> Tuple[np.ndarray, np.ndarray]:
    """Rank the indices by the distance to the metal."""
    idx = np.array(idx)
    # ToDo: should use the path length
    # in this case, it simply means if there is one
    # direct single bond
    distances = -np.array([has_bond_to_metal(mof, i) for i in idx]).astype(int)

    minimum_distance = np.min(distances)
    return np.where(distances == minimum_distance)[0], np.where(distances != minimum_distance)[0]


def cluster_nodes(
    graph: nx.Graph, original_indices: List[int], mof: "mof.MOF"  # noqa: F821 - forward reference
) -> List[int]:
    g = nx.Graph(graph).to_undirected()
    terminal_nodes = []
    for node in g.nodes:
        if g.degree[node] == 1:
            terminal_nodes.append(node)
        if g.degree[node] > 2:
            raise ValueError("Unsupported connectivity of branching points.")

    if len(terminal_nodes) > 2:
        raise ValueError("Unsupported connectivity of branching points.")

    path = nx.shortest_path(g, terminal_nodes[0], terminal_nodes[1])
    # if odd return the middle node
    if len(path) % 2 == 1:
        return [original_indices[path[int(len(path) / 2)]]]

    return original_indices


def get_branch_points(mof: "mof.MOF") -> List[int]:  # noqa: F821 - forward reference
    """Get all branching points in the MOF.

    Args:
        mof (MOF): MOF object.

    Returns:
        List[int]: List of indices of branching points.
    """
    branch_points = [i for i in range(len(mof.structure)) if _is_branch_point(mof, i)]
    return filter_branch_points(mof, branch_points)
