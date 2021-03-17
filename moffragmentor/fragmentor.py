# -*- coding: utf-8 -*-
"""Some pure functions that are used to perform the node identification
Node classification techniques described in https://pubs.acs.org/doi/pdf/10.1021/acs.cgd.8b00126
"""
import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import List, Tuple

import networkx as nx
from pymatgen import Molecule
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph


def has_path_to_any_other_metal(mof, index: int, this_metal_index: int) -> bool:
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


def classify_neighbors(mof, node_atoms: List[int]) -> OrderedDict:
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


def fragment_all_node(mof, filter_out_solvent: bool = True) -> OrderedDict:
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
        if mof._is_branch_point(path[-1]):
            connection_index.add(path[-1])
        node_atoms.update(path)

    return OrderedDict(
        [
            ("node_indices", node_atoms),
            ("solvent_connections", solvent_filtered["solvent_connections"]),
            ("solvent_indices", solvent_filtered["solvent_indices"]),
            ("connecting_node_indices", connection_index),
        ]
    )


def has_two_metals_as_neighbor(mof, site_index):
    neighbors = mof.structure_graph.get_neighboring_sites(site_index)

    metal_neighbors = 0
    for neighbor in neighbors:
        if neighbor.species.is_metal:
            metal_neighbors += 1

    return metal_neighbors > 0


def fragment_oxo_node(mof, filter_out_solvent: bool = True) -> OrderedDict:
    solvent_filtered = classify_neighbors(mof, mof.metal_indices)
    node_atoms = set(mof.metal_indices)
    if filter_out_solvent:
        good_connections = node_atoms
    else:
        good_connections = node_atoms | set(
            sum(solvent_filtered["solvent_indices"], [])
        )

    # ToDo: fix me -> i should also include the atoms that are between to metals
    # This should also find formate

    return OrderedDict(
        [
            ("node_indices", good_connections),
            ("solvent_connections", solvent_filtered["solvent_connections"]),
            ("solvent_indices", solvent_filtered["solvent_indices"]),
            ("connecting_node_indices", mof.metal_indices),
        ]
    )


def is_valid_node(mof, node_indices: List[int]) -> Tuple[bool, List[int]]:
    """
    This function realizes that a node with only one atom is kind of unusal.

    Args:
        mof (MOF): instance of a MOF
        node_indices (List[int]): indices that were proposed to belong to a MOF

    Returns:
        tuple(bool, List[int]): If the node is valid this will be True, if the node is invalid, this will be false and the indices should probably be put back
    """
    if len(node_indices) > 1:
        return True, []

    warnings.warn(
        "Only one metal detected in a potential node. This is unusual and might indicate a bug, a ZIF or metal in a ligand"
    )

    supergraph = mof.structure_graph * (3, 3, 3)
    supergraph = supergraph.remove_nodes(node_indices)
    connected_components = nx.connected_components(supergraph)
    if connected_components > 1:
        warnings.warn(
            "The structure at hand is probably ZIF or other coordination network for which support is currently not implemented"
        )
        return False, []
    else:
        warnings.warn("The structure probably contains a metal atom in the linker.")
        return False, node_indices


def is_valid_linker(linker_molecule: Molecule) -> bool:
    """A valid linker has more than one atom and at least two
    connection points

    Args:
        linker_molecule (Molecule): pymatgen molecule instance

    Returns:
        bool: true if the linker is a valid linker according to
            this definition
    """
    if len(linker_molecule) < 2:
        return False

    number_connection_sites = 0

    for linker_index in range(len(linker_molecule)):
        if linker_molecule[linker_index].properties == {"binding": True}:
            number_connection_sites += 1

    return number_connection_sites >= 2


def get_subgraphs_as_molecules(structure_graph: StructureGraph, use_weights=False):
    """Copied from
    http://pymatgen.org/_modules/pymatgen/analysis/graphs.html#StructureGraph.get_subgraphs_as_molecules
    and removed the duplicate check
    Args:
        structure_graph ( pymatgen.analysis.graphs.StructureGraph): Structuregraph
    Returns:
        List: list of molecules
    """
    # creating a supercell is an easy way to extract
    # molecules (and not, e.g., layers of a 2D crystal)
    # without adding extra logic
    supercell_sg = structure_graph * (3, 3, 3)

    # make undirected to find connected subgraphs
    supercell_sg.graph = nx.Graph(supercell_sg.graph)

    # find subgraphs
    all_subgraphs = [
        supercell_sg.graph.subgraph(c)
        for c in nx.connected_components(supercell_sg.graph)
    ]

    # discount subgraphs that lie across *supercell* boundaries
    # these will subgraphs representing crystals
    molecule_subgraphs = []

    for subgraph in all_subgraphs:
        intersects_boundary = any(
            [d["to_jimage"] != (0, 0, 0) for u, v, d in subgraph.edges(data=True)]
        )
        if not intersects_boundary:
            molecule_subgraphs.append(nx.MultiDiGraph(subgraph))
    # add specie names to graph to be able to test for isomorphism
    for subgraph in molecule_subgraphs:
        for node in subgraph:
            subgraph.add_node(node, specie=str(supercell_sg.structure[node].specie))

    unique_subgraphs = []

    def node_match(n1, n2):
        return n1["specie"] == n2["specie"]

    def edge_match(e1, e2):
        if use_weights:
            return e1["weight"] == e2["weight"]
        else:
            return True

    for subgraph in molecule_subgraphs:
        already_present = [
            nx.is_isomorphic(subgraph, g, node_match=node_match, edge_match=edge_match)
            for g in unique_subgraphs
        ]

        if not any(already_present):
            unique_subgraphs.append(subgraph)

    def make_mols(molecule_subgraphs=molecule_subgraphs, center=False):
        molecules = []
        indices = []
        for subgraph in molecule_subgraphs:
            coords = [supercell_sg.structure[n].coords for n in subgraph.nodes()]
            species = [supercell_sg.structure[n].specie for n in subgraph.nodes()]
            binding = [
                supercell_sg.structure[n].properties["binding"]
                for n in subgraph.nodes()
            ]
            idx = [n for n in subgraph.nodes()]
            molecule = Molecule(species, coords, site_properties={"binding": binding})

            # shift so origin is at center of mass
            if center:
                molecule = molecule.get_centered_molecule()
            indices.append(idx)
            molecules.append(molecule)
        return molecules, indices

    #     molecules, indices = make_mols(molecule_subgraphs)
    molecules_unique, idx = make_mols(unique_subgraphs, center=True)

    def relabel_graph(multigraph):
        mapping = dict(zip(multigraph, range(0, len(multigraph.nodes()))))
        return nx.readwrite.json_graph.adjacency_data(
            nx.relabel_nodes(multigraph, mapping)
        )

    return (
        molecules_unique,
        [
            MoleculeGraph(mol, relabel_graph(graph))
            for mol, graph in zip(molecules_unique, unique_subgraphs)
        ],
        idx,
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


def find_node_clusters(mof):
    paths = []
    branch_sites = []

    connecting_paths_ = set()

    for metal_index in mof.metal_indices:
        p, b = recursive_dfs_until_branch(mof, metal_index, [], [])
        paths.append(p)
        branch_sites.append(b)

    g = _to_graph(paths)
    nodes = list(nx.connected_components(g))

    for metal, branch_sites_for_metal in zip(mof.metal_indices, branch_sites):
        for branch_site in branch_sites_for_metal:
            connecting_paths_.update(
                sum(
                    nx.all_shortest_paths(mof._undirected_graph, metal, branch_site), []
                )
            )

    bs = set(sum(branch_sites, []))
    connecting_paths_ -= set(mof.metal_indices)
    connecting_paths_ -= bs

    return nodes, bs, connecting_paths_
