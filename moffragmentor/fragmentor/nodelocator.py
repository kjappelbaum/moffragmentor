# -*- coding: utf-8 -*-
"""Some pure functions that are used to perform the node identification
Node classification techniques described in https://pubs.acs.org/doi/pdf/10.1021/acs.cgd.8b00126.

Note that we currently only place one vertex for every linker which might loose some
information about isomers
"""
from collections import namedtuple

import networkx as nx

from ..sbu import Node, NodeCollection
from ._graphsearch import _complete_graph, _to_graph, recursive_dfs_until_branch
from .filter import filter_nodes

__all__ = [
    "find_node_clusters",
    "create_node_collection",
    "NodelocationResult",
]

NodelocationResult = namedtuple(
    "NodelocationResult", ["nodes", "branching_indices", "connecting_paths"]
)


def find_node_clusters(mof) -> NodelocationResult:
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

    # From every metal index in the structure perform DFS up to a
    # branch point
    for metal_index in mof.metal_indices:
        p, b = recursive_dfs_until_branch(mof, metal_index, [], [])
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
    connecting_paths_ -= set(mof.metal_indices)
    connecting_paths_ -= bs

    res = NodelocationResult(nodes, bs, connecting_paths_)
    return res


def create_node_collection(
    mof, node_location_result: NodelocationResult
) -> NodeCollection:
    # ToDo: This is a bit indirect, it would be better if we would have a list of dicts to loop over
    nodes = []
    for i in range(len(node_location_result.nodes)):
        node_indices = node_location_result.nodes[i]
        node = Node.from_mof_and_indices(
            mof,
            node_indices,
            node_location_result.branching_indices & node_indices,
            node_location_result.connecting_paths & node_indices,
        )
        nodes.append(node)

    return NodeCollection(nodes)
