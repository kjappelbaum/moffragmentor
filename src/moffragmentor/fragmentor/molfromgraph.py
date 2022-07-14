# -*- coding: utf-8 -*-
"""Generate molecules as the subgraphs from graphs"""
from collections import defaultdict
from typing import Iterable, List, Tuple

import networkx as nx
import numpy as np
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.core import Element, Molecule, Site


def _is_in_cell(frac_coords: np.ndarray) -> bool:
    return (frac_coords <= 1).all()


def _is_any_atom_in_cell(frac_coords: np.ndarray) -> bool:
    return any(_is_in_cell(row) for row in frac_coords)


def _select_parts_in_cell(  # pylint:disable=too-many-arguments,too-many-locals
    molecules: List[Molecule],
    graphs: List[MoleculeGraph],
    indices: List[List[int]],
    indices_here: List[List[int]],
    centers: List[np.ndarray],
    fractional_coordinates: np.ndarray,
    coordinates: np.ndarray,
) -> Tuple[List[Molecule], List[MoleculeGraph], List[List[int]]]:
    valid_indices = defaultdict(list)
    for i, ind in enumerate(indices_here):
        # change this check to having an atom in the cell
        frac_coords = fractional_coordinates[ind]

        if _is_any_atom_in_cell(frac_coords):
            sorted_idx = sorted(indices[i])
            valid_indices[str(sorted_idx)].append(i)

    molecules_ = []
    selected_indices = []
    graphs_ = []
    centers_ = []
    coordinates_ = []

    for _, v in valid_indices.items():
        for index in v:
            selected_indices.append(indices[index])
            molecules_.append(molecules[index])
            graphs_.append(graphs[index])
            centers_.append(centers[index])
            coordinates_.append(coordinates[index])

    return molecules_, graphs_, selected_indices, centers_, coordinates_


def get_mass(atomic_symbol):

    elem = Element(atomic_symbol)
    return elem.atomic_mass


def com(xyz, mass):
    mass = mass.reshape((-1, 1))
    return (xyz * mass).mean(0)


def get_subgraphs_as_molecules(  # pylint:disable=too-many-locals
    structure_graph: StructureGraph,
    use_weights: bool = False,
    return_unique: bool = True,
    disable_boundary_crossing_check: bool = False,
    filter_in_cell: bool = True,
) -> Tuple[
    List[Molecule], List[MoleculeGraph], List[List[int]], List[np.ndarray], List[np.ndarray]
]:
    """Isolates connected components as molecules from a StructureGraph.

    Copied from http://pymatgen.org/_modules/pymatgen/analysis/graphs.html#StructureGraph.get_subgraphs_as_molecules
    and removed the duplicate check

    Args:
        structure_graph (StructureGraph): Structuregraph
        use_weights (bool): If True, use weights for the edge matching
        return_unique (bool): If true, it only returns the unique molecules.
            If False, it will return all molecules that
            are completely included in the unit cell
            and fragments of the ones that are only partly in the cell
        disable_boundary_crossing_check (bool): If true, it will not check
            if the molecules are crossing the boundary of the unit cell.
            Default is False.
        filter_in_cell (bool): If True, it will only return molecules
            that have at least one atom in the cell

    Returns:
        Tuple[List[Molecule], List[MoleculeGraph], List[List[int]], List[np.ndarray], List[np.ndarray]]:
            A tuple of (molecules, graphs, indices, centers, coordinates)
    """
    # creating a supercell is an easy way to extract
    # molecules (and not, e.g., layers of a 2D crystal)
    # without adding extra logic
    supercell_sg = structure_graph * (3, 3, 3)

    # make undirected to find connected subgraphs
    supercell_sg.graph = nx.Graph(supercell_sg.graph)

    # find subgraphs
    all_subgraphs = [
        supercell_sg.graph.subgraph(c).copy() for c in nx.connected_components(supercell_sg.graph)
    ]

    # discount subgraphs that lie across *supercell* boundaries
    # these will subgraphs representing crystals
    molecule_subgraphs = []

    for subgraph in all_subgraphs:
        if disable_boundary_crossing_check:
            molecule_subgraphs.append(nx.MultiDiGraph(subgraph))
        else:
            intersects_boundary = any(
                (d["to_jimage"] != (0, 0, 0) for u, v, d in subgraph.edges(data=True))
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
        return True

    if return_unique:
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
        indices_here = []
        mol_centers = []
        coordinates = []
        for subgraph in molecule_subgraphs:
            coords = [supercell_sg.structure[n].coords for n in subgraph.nodes()]
            species = [supercell_sg.structure[n].specie for n in subgraph.nodes()]
            idx = [subgraph.nodes[n]["idx"] for n in subgraph.nodes()]
            idx_here = list(subgraph.nodes())
            molecule = Molecule(species, coords)  # site_properties={"binding": binding}

            masses = np.array(
                [get_mass(str(supercell_sg.structure[idx].specie)) for idx in idx_here]
            )
            mol_centers.append(com(supercell_sg.structure.cart_coords[idx_here], masses))
            # shift so origin is at center of mass
            if center:
                molecule = molecule.get_centered_molecule()
            indices.append(idx)
            molecules.append(molecule)
            indices_here.append(idx_here)
            coordinates.append(coords)
        return molecules, indices, indices_here, mol_centers, coordinates

    def relabel_graph(multigraph):
        mapping = dict(zip(multigraph, range(0, len(multigraph.nodes()))))
        return nx.readwrite.json_graph.adjacency_data(nx.relabel_nodes(multigraph, mapping))

    if return_unique:
        mol, idx, indices_here, centers, coordinates = make_mols(unique_subgraphs, center=True)
        return_subgraphs = unique_subgraphs
        return (
            mol,
            [MoleculeGraph(mol, relabel_graph(graph)) for mol, graph in zip(mol, return_subgraphs)],
            idx,
            centers,
            coordinates,
        )

    mol, idx, indices_here, centers, coordinates = make_mols(molecule_subgraphs)

    return_subgraphs = [
        MoleculeGraph(mol, relabel_graph(graph)) for mol, graph in zip(mol, molecule_subgraphs)
    ]

    if filter_in_cell:
        mol, return_subgraphs, idx, centers, coordinates = _select_parts_in_cell(
            mol,
            return_subgraphs,
            idx,
            indices_here,
            centers,
            structure_graph.structure.lattice.get_fractional_coords(
                supercell_sg.structure.cart_coords
            ),
            coordinates,
        )

    return mol, return_subgraphs, idx, centers, coordinates


def wrap_molecule(
    mol_idxs: Iterable[int], mof: "MOF", starting_index: int = 0
) -> Molecule:  # noqa: F821
    """Wrap a molecule in the cell of the MOF by walking along the structure graph.

    For this we perform BFS from the starting index. That is, we use a queue to
    keep track of the indices of the atoms that we still need to visit
    (the neighbors of the current index).
    We then compute new coordinates by computing the Cartesian coordinates
    of the neighbor image closest to the new coordinates of the current atom.

    To then create a Molecule with the correct ordering of sites, we walk
    through the hash table in the order of the original indices.

    Args:
        mol_idxs (Iterable[int]): The indices of the atoms in the molecule in the MOF.
        mof (MOF): MOF object that contains the mol_idxs.
        starting_index (int, optional): Starting index for the walk.
            Defaults to 0.

    Returns:
        Molecule: wrapped molecule
    """
    new_positions_cart = {}
    new_positions_frac = {}
    still_to_wrap_queue = [mol_idxs[starting_index]]
    new_positions_cart[mol_idxs[starting_index]] = mof.cart_coords[mol_idxs[starting_index]]
    new_positions_frac[mol_idxs[starting_index]] = mof.frac_coords[mol_idxs[starting_index]]

    while still_to_wrap_queue:
        current_index = still_to_wrap_queue.pop(0)
        if current_index in mol_idxs:
            neighbor_indices = mof.get_neighbor_indices(current_index)
            for neighbor_index in neighbor_indices:
                if (neighbor_index not in new_positions_cart) & (neighbor_index in mol_idxs):
                    _, image = mof.structure[neighbor_index].distance_and_image_from_frac_coords(
                        new_positions_frac[current_index]
                    )
                    new_positions_frac[neighbor_index] = mof.frac_coords[neighbor_index] - image
                    new_positions_cart[neighbor_index] = mof.lattice.get_cartesian_coords(
                        new_positions_frac[neighbor_index]
                    )
                    still_to_wrap_queue.append(neighbor_index)

    new_sites = []
    for idx in mol_idxs:
        new_sites.append(Site(mof.structure[idx].species, new_positions_cart[idx]))

    return Molecule.from_sites(new_sites)
