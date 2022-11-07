# -*- coding: utf-8 -*-
"""Some pure functions that are used to perform the node identification.

Node classification techniques described
in https://pubs.acs.org/doi/pdf/10.1021/acs.cgd.8b00126.
"""
from collections import namedtuple
from typing import Iterable, List, Optional

import networkx as nx
from loguru import logger
from skspatial.objects import Points

from moffragmentor.descriptors.sbu_dimensionality import get_sbu_dimensionality
from moffragmentor.fragmentor._graphsearch import (
    _complete_graph,
    _to_graph,
    recursive_dfs_until_branch,
    recursive_dfs_until_cn3,
)
from moffragmentor.sbu import Node, NodeCollection
from moffragmentor.utils import _flatten_list_of_sets, _get_metal_sublist

__all__ = [
    "find_node_clusters",
    "create_node_collection",
    "NodelocationResult",
]

NodelocationResult = namedtuple(
    "NodelocationResult",
    [
        "nodes",
        "branching_indices",
        "connecting_paths",
        "binding_indices",
        "to_terminal_from_branching",
    ],
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


def find_node_clusters(
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
    paths, to_terminal_from_branching = _complete_graph(mof, paths, branch_sites)
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
                        for neighbor in mof.get_neighbor_indices(site):
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

    res = NodelocationResult(
        nodes, bs, connecting_paths_, binding_sites, to_terminal_from_branching
    )
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
            branching_indices=node_location_result.branching_indices,
            binding_indices=node_location_result.binding_indices,
        )
        nodes.append(node)

    return NodeCollection(nodes)


def break_rod_node(mof, indices):
    metal_subset = {i for i in indices if i in mof.metal_indices}
    if not isinstance(metal_subset, int):
        return [{i} for i in metal_subset]
    else:
        return [{metal_subset}]


def create_single_metal_nodes(mof, node_result):
    new_nodes = []
    for node in node_result.nodes:
        new_nodes.extend(break_rod_node(mof, node))
    new_node_result = NodelocationResult(
        new_nodes,
        node_result.branching_indices,
        node_result.connecting_paths,
        node_result.binding_indices,
        node_result.to_terminal_from_branching,
    )
    return new_node_result


def break_rod_nodes(mof, node_result):
    """Break rod nodes into smaller pieces."""
    new_nodes = []
    for node in node_result.nodes:
        if get_sbu_dimensionality(mof, node) >= 1:
            logger.debug("Found 1- or 2-dimensional node. Will break into isolated metals.")
            new_nodes.extend(break_rod_node(mof, node))
        else:
            new_nodes.append(node)
    new_node_result = NodelocationResult(
        new_nodes,
        node_result.branching_indices,
        node_result.connecting_paths,
        node_result.binding_indices,
        node_result.to_terminal_from_branching,
    )
    return new_node_result


def check_node(node_indices, branching_indices, mof) -> bool:
    """Check if the node seems to be a reasonable SBU.

    If not, we can use this to change the fragmentation to something more robust.

    Args:
        node_indices (set): set of indices that make up the node
        branching_indices (set): set of indices that are branching indices
        mof (MOF): MOF object

    Returns:
        bool: True if the node is reasonable, False otherwise
    """
    # check if there is not way more organic than metal
    num_organic = len(node_indices & set(mof.c_indices)) + len(node_indices & set(mof.n_indices))
    num_metals = len(node_indices & set(mof.metal_indices))
    branching_indices_in_node = branching_indices & node_indices
    if num_organic > num_metals + len(branching_indices_in_node):
        return False
    return True


def break_organic_nodes(node_result, mof):
    """If we have a node that is mostly organic, we break it up into smaller pieces."""
    new_nodes = []
    for node in node_result.nodes:
        if len(node) == len(mof):
            logger.debug("Breaking node as full MOF is assigned as node.")
            new_nodes.extend(break_rod_node(mof, node))
        elif check_node(node, node_result.branching_indices, mof) or might_be_porphyrin(
            node, node_result.branching_indices, mof
        ):
            new_nodes.append(node)
        else:
            logger.debug(
                f"Found node {node} that is mostly organic. Will break into isolated metals."
            )
            new_nodes.extend(break_rod_node(mof, node))
    return NodelocationResult(
        new_nodes,
        node_result.branching_indices,
        node_result.connecting_paths,
        node_result.binding_indices,
        node_result.to_terminal_from_branching,
    )


def find_nodes(
    mof,
    unbound_solvent: "NonSbuMoleculeCollection" = None,  # noqa: F821
    forbidden_indices: Optional[Iterable[int]] = None,
    create_single_metal_bus: bool = False,
    check_dimensionality: bool = True,
    break_organic_nodes_: bool = True,
) -> NodeCollection:
    """Find the nodes in a MOF.

    Args:
        mof (MOF): moffragmentor MOF instance
        unbound_solvent (NonSbuMoleculeCollection): collection of unbound solvent molecules
        forbidden_indices (Optional[Iterable[int]]): indices of sites that should not be considered
        create_single_metal_bus (bool): if True, single metal nodes will be created
        check_dimensionality (bool): if True, rod nodes will be broken into single metal nodes
        break_organic_nodes_ (bool): if True, rod nodes will be broken into single metal nodes
            if they are mostly organic

    Returns:
        NodeCollection: collection of nodes
    """
    if forbidden_indices is None:
        forbidden_indices = []

    node_result = find_node_clusters(
        mof, unbound_solvent.indices, forbidden_indices=forbidden_indices
    )
    if create_single_metal_bus:
        # Rewrite the node result
        node_result = create_single_metal_nodes(mof, node_result)
    if check_dimensionality:
        # If we have nodes with dimensionality >0, we change these nodes to only contain the metal
        node_result = break_rod_nodes(mof, node_result)
    if break_organic_nodes_:
        # ToDo: This, of course, would also break prophyrin ...
        node_result = break_organic_nodes(node_result, mof)

    node_collection = create_node_collection(mof, node_result)

    return node_result, node_collection


def metal_and_branching_coplanar(node_idx, all_branching_idx, mof, tol=0.1):
    branching_idx = list(node_idx & all_branching_idx)
    coords = mof.frac_coords[list(node_idx) + branching_idx]
    points = Points(coords)
    return points.are_coplanar(tol=tol)


def might_be_porphyrin(node_indices, branching_idx, mof):
    metal_in_node = _get_metal_sublist(node_indices, mof.metal_indices)
    node_indices = set(node_indices)
    branching_idx = set(branching_idx)
    bound_to_metal = sum([mof.get_neighbor_indices(i) for i in metal_in_node], [])
    branching_bound_to_metal = branching_idx & set(bound_to_metal)
    # ToDo: check and think if this can handle the general case
    # it should, at least if we only look at the metals
    if (len(metal_in_node) == 1) & (len(node_indices | branching_bound_to_metal) > 1):
        logger.debug(
            "metal_in_node",
            metal_in_node,
            node_indices,
        )
        num_neighbors = len(bound_to_metal)
        if (
            metal_and_branching_coplanar(node_indices, branching_bound_to_metal, mof)
            & (num_neighbors > 2)
            & (len(branching_bound_to_metal) < 5)
        ):
            logger.debug(
                "Metal in linker found, indices: {}".format(
                    node_indices,
                )
            )
            return True

    return False


def detect_porphyrin(node_collection, mof):
    not_node = []
    for i, node in enumerate(node_collection):
        if might_be_porphyrin(node._original_indices, node._original_graph_branching_indices, mof):
            logger.info("Found porphyrin in node {}".format(node._original_indices))
            not_node.append(i)
    return not_node
