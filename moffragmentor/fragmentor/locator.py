# -*- coding: utf-8 -*-
"""Some pure functions that are used to perform the node identification
Node classification techniques described in https://pubs.acs.org/doi/pdf/10.1021/acs.cgd.8b00126
"""
from collections import OrderedDict, namedtuple
from copy import deepcopy
from typing import List, Set

import networkx as nx

from ..molecule import NonSbuMolecule, NonSbuMoleculeCollection
from ..sbu import Linker, LinkerCollection, Node, NodeCollection
from ..utils import _not_relevant_structure_indices
from ..utils.errors import NoMetalError
from .filter import filter_nodes
from .splitter import get_subgraphs_as_molecules
from .utils import _get_metal_sublist

__all__ = [
    "find_node_clusters",
    "create_node_collection",
    "get_all_bound_solvent_molecules",
    "create_linker_collection",
]

Nodelocation_Result = namedtuple(
    "Nodelocation_Result", ["nodes", "branching_indices", "connecting_paths"]
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
    g = deepcopy(mof.nx_graph)
    g.remove_node(this_metal_index)
    for metal_index in metal_indices:
        if nx.has_path(g, index, metal_index):
            return True
    return False


def recursive_dfs_until_terminal(
    mof, start: int, path: List[int] = [], skip_list: List[int] = []
) -> List[int]:
    """From a given starting point perform depth-first search until leaf nodes are reached

    Args:
        mof (MOF): A MOF instance
        start (int): Starting index for the search
        path (List[int], optional): Starting path. Defaults to [].

    Returns:
        List[int]: Path between start and the leaf node
    """
    if (start not in path) and (start not in skip_list):
        path.append(start)
        if mof._is_terminal(start):
            return path

        for neighbour in mof.get_neighbor_indices(start):
            path = recursive_dfs_until_terminal(mof, neighbour, path, skip_list)

    return path


def _complete_graph(
    mof, paths: List[List[int]], branching_nodes: List[List[int]]
) -> List[List[int]]:
    """Loop over all paths that were traversed in DFS
    and then, if the verices are not branching add all the paths
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
            # ToDo: we can at least leverage the information about the bridges we have and not just brute-force the search for all vertices
            if vertex not in visited:
                p = recursive_dfs_until_terminal(mof, vertex, [], branching_nodes)
                subpath.extend(p)
        completed_edges.append(subpath + path)
    return completed_edges


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


def _locate_bound_solvent(mof, node_atoms: Set[int]) -> OrderedDict:
    solvent_connections = set()

    solvent_indices = []
    good_connections = set()

    metal_subset = _get_metal_sublist(node_atoms, mof.metal_indices)
    for metal_index in metal_subset:
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


def _get_solvent_molecules_bound_to_node(
    mof, node_atoms: Set[int]
) -> NonSbuMoleculeCollection:
    """Locate solvent molecules bound to one MOF node.
    Bound solvent is defined as being connected via one bridge
    to one metal center.

    Args:
        mof (MOF): Instance of a MOF object
        node_atoms (Set[int]): Indices that were identified as
            belonging to a node cluster.

    Returns:
        NonSbuMoleculeCollection: Collection of NonSbuMolecule objects
            containing the bound solvent molecules
    """
    bound_solvent_location_result = _locate_bound_solvent(mof, node_atoms)

    molecules = []

    for solvent_ind in bound_solvent_location_result["solvent_indices"]:
        molecules.append(
            NonSbuMolecule.from_structure_graph_and_indices(
                mof.structure_graph, solvent_ind
            )
        )

    return NonSbuMoleculeCollection(molecules)


def get_all_bound_solvent_molecules(
    mof, node_atom_sets: List[Set[int]]
) -> NonSbuMoleculeCollection:
    """Given a MOF object and a collection of node atoms, identify all bound solvent molecules.
    Bound solvent is defined as being connected via one bridge
    to one metal center.

    Args:
        mof (MOF): instance of a MOF object
        node_atom_sets (List[Set[int]]): List of indices for the MOF nodes

    Returns:
        NonSbuMoleculeCollection: Collection of NonSbuMolecule objects
            containing the bound solvent molecules
    """
    non_sbu_molecule_collections = NonSbuMoleculeCollection([])
    for node_atom_set in node_atom_sets:
        non_sbu_molecule_collections += _get_solvent_molecules_bound_to_node(
            mof, node_atom_set
        )

    return non_sbu_molecule_collections


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


def find_node_clusters(mof) -> Nodelocation_Result:
    """This function locates the branchin indices, and node clusters in MOFs.
    Starting from the metal indices it performs depth first search on the structure
    graph up to branching points.

    Args:
        mof (MOF): moffragmentor MOF instance

    Returns:
        Nodelocation_Result: nametuple with the slots "nodes", "branching_indices" and
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
    g = _to_graph(paths)
    nodes = list(nx.connected_components(g))
    # filter out "node" candidates that are not actual nodes.
    # in practice this is relevant for ligands with metals in them (e.g., porphyrins)
    nodes = filter_nodes(nodes, mof.structure_graph, mof.metal_indices)

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

    res = Nodelocation_Result(nodes, bs, connecting_paths_)
    return res


def create_node_collection(
    mof, node_location_result: Nodelocation_Result
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


def _create_linkers_from_node_location_result(
    mof, node_location_result, unbound_solvent
):
    linkers = []

    all_node_indices = set()
    for node_indices in node_location_result.nodes:
        all_node_indices.update(node_indices)

    not_linker_indices = (
        all_node_indices
        - node_location_result.connecting_paths
        - node_location_result.branching_indices
    ) | set(unbound_solvent.indices)

    graph_ = deepcopy(mof.structure_graph)
    graph_.remove_nodes(not_linker_indices)
    mols, graphs, idxs = get_subgraphs_as_molecules(
        graph_, return_unique=False, original_len=len(mof)
    )

    for mol, graph, idx in zip(mols, graphs, idxs):
        idxs = set(idx)
        linker = Linker(
            mol,
            graph,
            node_location_result.branching_indices & idxs,
            node_location_result.connecting_paths & idxs,
            idx,
        )
        linkers.append(linker)
    return linkers


def create_linker_collection(
    mof, node_location_result: Nodelocation_Result, unbound_solvents
) -> LinkerCollection:
    linkers = _create_linkers_from_node_location_result(
        mof, node_location_result, unbound_solvents
    )
    return LinkerCollection(linkers)
