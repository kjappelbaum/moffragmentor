# -*- coding: utf-8 -*-
"""Some pure functions that are used to perform the node identification
Node classification techniques described in https://pubs.acs.org/doi/pdf/10.1021/acs.cgd.8b00126.

Note that we currently only place one vertex for every linker which might loose some
information about isomers
"""
from collections import OrderedDict, defaultdict, namedtuple
from copy import deepcopy
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np

from ..molecule import NonSbuMolecule, NonSbuMoleculeCollection
from ..sbu import Linker, LinkerCollection, Node, NodeCollection
from ..utils import _flatten_list_of_sets
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


def _to_graph(mof, paths, branch_sites):
    """https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements"""
    G = nx.Graph()
    for part in paths:
        G.add_nodes_from(part)
        G.add_edges_from(_to_edges(part))
    G.add_edges_from(
        _connect_connected_branching_indices(mof, _flatten_list_of_sets(branch_sites))
    )
    return G


def _to_edges(paths):
    it = iter(paths)
    last = next(it)

    for current in it:
        yield last, current
        last = current


def _connect_connected_branching_indices(mof, flattend_path):
    edges = set()

    for i in flattend_path:
        neighbors = mof.get_neighbor_indices(i)
        for neighbor in neighbors:
            if neighbor in flattend_path:
                edges.add(tuple(sorted((i, neighbor))))

    return edges


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
    g = _to_graph(mof, paths, branch_sites)
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


def _pick_linker_indices(
    idxs: List[List[int]], centers, all_node_branching_indices
) -> Tuple[List[int], List[int]]:
    """Trying to have a more reasonable way to filter out linkers
    (of multiple versions of the linker that might be wrapped across a unit cell)"""
    counter = 0
    unique_branching_site_centers = {}
    unique_branching_sites_indices = {}
    counter = 0
    has_branch_point = []
    for idx, center in zip(idxs, centers):
        intersection = set(idx) & all_node_branching_indices
        if len(intersection) >= 2:
            has_branch_point.append(counter)
            intersection = tuple(sorted(tuple(intersection)))
            norm = np.linalg.norm(center)
            if intersection in unique_branching_site_centers.keys():
                if unique_branching_site_centers[intersection] > norm:
                    unique_branching_site_centers[intersection] = norm
                    unique_branching_sites_indices[intersection] = counter
            else:
                unique_branching_site_centers[intersection] = norm
                unique_branching_sites_indices[intersection] = counter
        counter += 1

    return unique_branching_sites_indices.values(), has_branch_point


def _get_exploded_node_neighbors_dict(
    node_collection: NodeCollection, idxs: List[int], centers: List[np.ndarray]
) -> Dict[int, dict]:
    """The purpose of this function is to create an edge dictionary of the form
    {node_index: {
        (branching_indices): [Linker, Linker]
    }}
    Where branching_indices is a sorted tuple of the indices which connect the node and linker and
    linker is a tuple with the linker index and the cartesian coordinates of the center
    The linkers here are unfiltered in the sense that if the len(list)>1 we also have periodic images.
    A subsequent function will pick on setting and find out what image the other one is.

    We need to go trough all these steps to have a clean net representation, which one then could
    also use for topological analysis.

    Args:
        node_collection (NodeCollection): nodes found in the MOF
        idxs (List[int]): Linker indices found after subgraph extraction
        centers (List[np.ndarray]): Linker centers in cartesian cooordinates

    Returns:
        Dict[int, dict]: Linkers neighboring a given edge
    """
    counter = 0
    node_neighbors = {}
    for i, node in enumerate(node_collection):
        exploded_edge_dict = defaultdict(list)
        for j, (idx, center) in enumerate(zip(idxs, centers)):
            overlap_w_branch_points = node.original_branching_indices & set(idx)
            if overlap_w_branch_points:
                exploded_edge_dict[
                    tuple(sorted(tuple(overlap_w_branch_points)))
                ].append((j, center))
                counter += 1
            node_neighbors[i] = exploded_edge_dict
    return node_neighbors


def _get_new_edge_dict(
    exploded_edge_dict: Dict[Tuple[int, int], List[Tuple[int, np.ndarray]]],
    mof: object,
    linker_indices: List[int],
) -> Dict[Tuple[int, int], List[Tuple[int, np.ndarray, np.ndarray]]]:
    """This function works on the neighbors of one node after we picked linker images we want to keep in the cell.
    That is, if for one branching point we currently have more than one linker center the other ones are periodic images.
    We will figure out what images we deal with and write a new edge dictionary

    Args:
        exploded_edge_dict (Dict[Tuple[int, int], List[Tuple[int, np.ndarray]]]): The neighbors of one node
        mof (MOF): An instance of the MOF class, we need it to get the lattice
        linker_indices (List[int]): List of linker indices (from the list of all linkers) we selected to be in the cell

    Returns:
        Dict[Tuple[int, int], List[Tuple[int, np.ndarray, np.ndarray]]]: Updated edge dict for one node.
        The elements of the tuple are the linker index, the image (if it is in the cell the image will be [0,0,0]),
         and the center (in Cartesian coordinates)
    """
    new_edge_dict = {}
    for key, value in exploded_edge_dict.items():
        new_list = []
        relevant_subindex = None
        for counter, (idx, center) in enumerate(value):
            if idx in linker_indices:
                relevant_subindex = counter
                break

        linker_index = [j for j, i in enumerate(linker_indices) if i == idx][0]

        selected_center = mof.lattice.get_fractional_coords(value[relevant_subindex][1])

        for counter, (idx, center) in enumerate(value):
            frac_coords_1 = mof.lattice.get_fractional_coords(center)

            _, image = mof.lattice.get_distance_and_image(
                selected_center, frac_coords_1
            )
            new_list.append((linker_index, image, center))
        new_edge_dict[key] = new_list

    return new_edge_dict


def _compress_node_neighbors_dict(
    node_neighbors: Dict[int, dict], mof: object, linker_indices: List[int]
) -> Dict[int, dict]:
    """The purpose of this function is to loop over all nodes and clean up the edge dict using the information
    which linkers we selected to be in the cell. The other ones will be indicated as periodic images.

    Args:
        node_neighbors (Dict[int, dict]): Linkers neighboring a given edge
        mof (MOF): Instance of the MOF class
        linker_indices (List[int]): linkers selected to be in the cell

    Returns:
        Dict[int, dict]: Updated node_neighbors dict. Note that the tuples for the linkers are now one element
            longer since they also contain the image
    """
    new_node_neighbors_dict = {}

    for node_index, node_edges in node_neighbors.items():
        new_node_neighbors_dict[node_index] = _get_new_edge_dict(
            node_edges, mof, linker_indices
        )

    return new_node_neighbors_dict


def _get_connected_linkers(
    mof, branching_coordinates: List[np.array], linker_collection
) -> List[Tuple[int, np.array, np.array]]:
    """The insight of this function is that the branching indices outside the cell a node might
    be bound to are periodic images of the ones in the cell"""
    linked_to = []
    for i, branching_coordinate in enumerate(branching_coordinates):
        frac_a = mof.lattice.get_fractional_coords(branching_coordinate)
        for j, linker in enumerate(linker_collection):
            for coord in linker.branching_coords:
                frac_b = mof.lattice.get_fractional_coords(coord)

                distance, image = mof.lattice.get_distance_and_image(frac_a, frac_b)
                if distance < 0.001:
                    center_frac = mof.lattice.get_fractional_coords(linker.center)

                    linked_to.append(
                        (
                            j,
                            image,
                            mof.lattice.get_cartesian_coords(center_frac + image),
                        )
                    )

    return linked_to


def _get_edge_dict(mof, node_collection, linker_collection):
    edge_dict = {}
    for i, node in enumerate(node_collection):
        edge_dict[i] = _get_connected_linkers(
            mof, node.branching_coords, linker_collection
        )

    return edge_dict


def _create_linkers_from_node_location_result(
    mof, node_location_result, node_collection, unbound_solvent
) -> Tuple[LinkerCollection, dict]:
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
    mols, graphs, idxs, centers, coordinates = get_subgraphs_as_molecules(
        graph_,
        return_unique=False,
        filter_in_cell=False,
        disable_boundary_crossing_check=True,
    )

    linker_indices, _ = _pick_linker_indices(
        idxs, centers, node_location_result.branching_indices
    )

    for i, (mol, graph, idx, center) in enumerate(zip(mols, graphs, idxs, centers)):
        idxs = set(idx)
        linker = Linker(
            mol,
            graph,
            center,
            node_location_result.branching_indices & idxs,
            node_location_result.connecting_paths & idxs,
            idx,
        )

        if i in linker_indices:
            linkers.append(linker)

    linker_collection = LinkerCollection(linkers)
    edge_dict = _get_edge_dict(mof, node_collection, linker_collection)
    return linker_collection, edge_dict


def create_linker_collection(
    mof,
    node_location_result: Nodelocation_Result,
    node_collection: NodeCollection,
    unbound_solvents,
) -> Tuple[LinkerCollection, dict]:
    linker_collection, edge_dict = _create_linkers_from_node_location_result(
        mof, node_location_result, node_collection, unbound_solvents
    )
    return linker_collection, edge_dict
