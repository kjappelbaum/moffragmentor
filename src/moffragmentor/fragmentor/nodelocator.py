# -*- coding: utf-8 -*-
"""Some pure functions that are used to perform the node identification
Node classification techniques described
in https://pubs.acs.org/doi/pdf/10.1021/acs.cgd.8b00126.

Note that we currently only place one vertex for every linker which might loose some
information about isomers
"""
from collections import namedtuple
from typing import List

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
    "NodelocationResult", ["nodes", "branching_indices", "connecting_paths"]
)


def find_node_clusters(  # pylint:disable=too-many-locals
    mof, unbound_solvent_indices=None
) -> NodelocationResult:
    """This function locates the branchin indices, and node clusters in MOFs.
    Starting from the metal indices it performs depth first search on the structure
    graph up to branching points.

    Args:
        mof (MOF): moffragmentor MOF instance

    Returns:
        NodelocationResult: nametuple with the slots "nodes", "branching_indices" and
            "connecting_paths"

    Raises:
        NoMetalError: In case the structure does not contain any metal. The presence of
            a metal is crucial for the fragmentation algorithm.
    """
    paths = []
    branch_sites = []

    connecting_paths_ = set()

    if unbound_solvent_indices is None:
        unbound_solvent_indices = []
    # From every metal index in the structure perform DFS up to a
    # branch point
    for metal_index in mof.metal_indices:
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

        for metal_index in mof.metal_indices:
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

    # filter out "node" candidates that are not actual nodes.
    # in practice this is relevant for ligands with metals in them (e.g., porphyrins)
    # nodes = filter_nodes(
    #     nodes, mof.structure_graph, mof.metal_indices, mof.terminal_indices
    # )

    bs = set(sum(branch_sites, []))

    # we store the shortest paths between nodes and branching indices
    # ToDo: we can extract this from the DFS paths above
    for metal, branch_sites_for_metal in zip(mof.metal_indices, branch_sites):
        for branch_site in branch_sites_for_metal:
            paths = list(nx.all_shortest_paths(mof.nx_graph, metal, branch_site))
            for p in paths:
                metal_in_path = [i for i in p if i in mof.metal_indices]
                if len(metal_in_path) == 1:
                    connecting_paths_.update(p)

    # from the connecting paths we remove the metal indices and the branching indices
    # we need to remove the metal indices as otherwise the fragmentation breaks
    connecting_paths_ -= set(mof.metal_indices)

    res = NodelocationResult(nodes, bs, connecting_paths_)
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
            binding_indices=identify_node_binding_indices(
                mof,
                node_indices,
                node_location_result.connecting_paths,
                node_location_result.branching_indices,
            ),
            connecting_paths=node_location_result.connecting_paths & node_indices,
        )
        nodes.append(node)

    return NodeCollection(nodes)


def identify_node_binding_indices(mof, indices, connecting_paths, binding_indices) -> List[int]:
    """For the metal clusters, our rule for binding indices is quite simple.
    We simply take the metal that is part of the connecting path.
    We then additionally filter based on the constraint that
    the nodes we want to identify need to bee bound to what
    we have in the connecting path
    """
    filtered = []
    candidates = set(mof.metal_indices) & set(indices)
    for candidate in candidates:
        if len(set(mof.get_neighbor_indices(candidate)) & connecting_paths | binding_indices):
            filtered.append(candidate)

    return filtered
