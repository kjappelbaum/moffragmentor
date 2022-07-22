# -*- coding: utf-8 -*-
"""Some pure functions that are used to perform the node identification.

Node classification techniques described
in https://pubs.acs.org/doi/pdf/10.1021/acs.cgd.8b00126.

Note that we currently only place one vertex for every linker which might loose some information about isomers
"""
from collections import namedtuple
from typing import List, Optional

import networkx as nx
from loguru import logger

from ._graphsearch import (
    _complete_graph,
    _to_graph,
    recursive_dfs_until_branch,
    recursive_dfs_until_cn3,
)
from ..sbu import Node, NodeCollection
from ..utils import _flatten_list_of_sets

__all__ = [
    "find_node_clusters",
    "create_node_collection",
    "NodelocationResult",
]

NodelocationResult = namedtuple(
    "NodelocationResult", ["nodes", "branching_indices", "connecting_paths", "binding_indices"]
)


def _count_metals_in_path(path, metal_indices):
    """Count the number of metals in a path.

    Args:
        path (List[int]): path to check
        metal_indices (List[int]): indices of metals

    Returns:
        int: number of metals in the path
    """
    return len(set(path) & set(metal_indices))


def _path_without_metal_and_branching_sites(path, metal_indices, branching_indices):
    """Remove the metal indices and the branching indices from a path.

    Args:
        path (List[int]): path to remove the indices from
        metal_indices (List[int]): indices of metals
        branching_indices (List[int]): indices of branching sites

    Returns:
        List[int]: path without metal indices and branching indices
    """
    return [i for i in path if i not in metal_indices and i not in branching_indices]


def find_node_clusters(  # pylint:disable=too-many-locals
    mof,
    unbound_solvent_indices: Optional[List[int]] = None,
    forbidden_indices: Optional[List[int]] = None,
) -> NodelocationResult:
    """Locate the branching indices, and node clusters in MOFs.

    Starting from the metal indices it performs depth first search
    on the structure graph up to branching points.

    Args:
        mof (MOF): moffragmentor MOF instance
        unbound_solvent_indices (List[int], optionl):
            indices of unbound solvent atoms. Defaults to None.
        forbidden_indices (List[int], optional):
            indices not considered as metals, for instance, because
            they are part of a linker. Defaults to None.

    Returns:
        NodelocationResult: nametuple with the slots "nodes", "branching_indices" and
            "connecting_paths"
    """
    logger.debug("Locating node clusters")
    paths = []
    branch_sites = []

    connecting_paths_ = set()

    if forbidden_indices is None:
        forbidden_indices = []

    metal_indices = set([i for i in mof.metal_indices if i not in forbidden_indices])

    if unbound_solvent_indices is None:
        unbound_solvent_indices = []
    # From every metal index in the structure perform DFS up to a
    # branch point
    for metal_index in metal_indices:
        if metal_index not in unbound_solvent_indices:
            p, b = recursive_dfs_until_branch(mof, metal_index, [], [])
            paths.append(p)
            branch_sites.append(b)
    if len(_flatten_list_of_sets(branch_sites)) == 0:
        logger.warning(
            "No branch sites found according to branch site definition.\
             Using now CN=3 sites between metals as branch sites.\
                This is not consistent with the conventions \
                    used in other parts of the code."
        )
        paths = []
        connecting_paths_ = set()
        branch_sites = []

        for metal_index in metal_indices:
            if metal_index not in unbound_solvent_indices:
                p, b = recursive_dfs_until_cn3(mof, metal_index, [], [])
                paths.append(p)
                branch_sites.append(b)

    # The complete_graph will add the "capping sites" like bridging OH
    # or capping formate
    paths = _complete_graph(mof, paths, branch_sites)

    # we find the connected components in those paths
    g = _to_graph(mof, paths, branch_sites)
    nodes = list(nx.connected_components(g))

    bs = set(sum(branch_sites, []))

    logger.debug("Locating binding sites and connecting paths")
    binding_sites = set()
    # we store the shortest paths between nodes and branching indices
    # ToDo: we can extract this from the DFS paths above
    for metal, branch_sites_for_metal in zip(metal_indices, branch_sites):
        for branch_site in branch_sites_for_metal:
            # note: using all_simple_paths here is computationally prohibitive
            paths = nx.all_shortest_paths(mof.nx_graph, metal, branch_site)
            for p in paths:
                metals_in_path = _count_metals_in_path(p, metal_indices)
                if metals_in_path >= 1:
                    connecting_paths_.update(p)
                if metals_in_path == 1:
                    binding_path = _path_without_metal_and_branching_sites(p, metal_indices, bs)
                    binding_sites.update(binding_path)
                    # traverse to also add things like Hs in the binding path
                    for site in binding_path:
                        for neighbor in g.neighbors(site):
                            if neighbor not in metal_indices | bs:
                                binding_sites.add(neighbor)

    all_neighbors = []
    for node in nodes:
        neighbors = mof.get_neighbor_indices(node)
        all_neighbors.extend(neighbors)
        intesection = set(neighbors) & bs
        if len(intesection) > 0:
            connecting_paths_.update(neighbors)

    # from the connecting paths we remove the metal indices and the branching indices
    # we need to remove the metal indices as otherwise the fragmentation breaks
    connecting_paths_ -= set(metal_indices)

    res = NodelocationResult(nodes, bs, connecting_paths_, binding_sites)
    return res


def create_node_collection(mof, node_location_result: NodelocationResult) -> NodeCollection:
    # ToDo: This is a bit indirect,
    # it would be better if we would have a list of dicts to loop over
    nodes = []
    for i, _ in enumerate(node_location_result.nodes):
        node_indices = node_location_result.nodes[i]
        node = Node.from_mof_and_indices(
            mof=mof,
            node_indices=node_indices,
            branching_indices=node_location_result.branching_indices & node_indices,
            binding_indices=node_location_result.binding_indices & node_indices,
            connecting_paths=node_location_result.connecting_paths & node_indices,
        )
        nodes.append(node)

    return NodeCollection(nodes)
