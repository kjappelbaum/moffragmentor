# -*- coding: utf-8 -*-
"""This module focusses on the extraction of pymatgen Molecules from a structure for which we know the branching points / node/linker indices"""
from copy import deepcopy
from typing import List, Tuple

import networkx as nx
from pymatgen import Molecule
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph

__all__ = ["get_subgraphs_as_molecules"]


def _filter_indices_in_cell(indices: List[List[int]], num_atoms: int) -> List[int]:
    """From a list of indices in a supercell return
    the indices of the sublists that are in the original cell.

    Args:
        indices (List[List[int]]): List of lists of indices
            (e.g. of molecules in a supercell)
        num_atoms (int): number of atoms in the original cell

    Returns:
        List[int]: Indices of the sublist that are in the original cell
    """
    selected_indices = []
    for i, index in enumerate(indices):
        if all([sub_index < num_atoms for sub_index in index]):
            selected_indices.append(i)
    return selected_indices


def _select_parts_in_cell(
    molecules: List[Molecule],
    graphs: List[MoleculeGraph],
    indices: List[List[int]],
    molecule_size: int,
) -> Tuple[List[Molecule], List[MoleculeGraph], List[List[int]]]:
    molecules_ = []
    graphs_ = []
    selected_indices = []
    to_delete = []
    for i, sublist in enumerate(indices):
        new_sublist = []
        delete_sublist = []
        for item in sublist:
            if item < molecule_size:
                new_sublist.append(item)
            else:
                delete_sublist.append(item)
        if new_sublist:
            selected_indices.append(new_sublist)
            to_delete.append(delete_sublist)
            molecules_.append(deepcopy(molecules[i]))
            graphs_.append(deepcopy(graphs[i]))

    for mol, graph, delete_sublist in zip(molecules_, graphs_, to_delete):
        mol.remove_sites(delete_sublist)
        graph.remove_nodes(delete_sublist)

    return molecules_, graphs_, selected_indices


def get_subgraphs_as_molecules(
    structure_graph: StructureGraph,
    use_weights: bool = False,
    return_unique: bool = True,
) -> Tuple[List[Molecule], List[MoleculeGraph], List[List[int]]]:
    """Copied from
    http://pymatgen.org/_modules/pymatgen/analysis/graphs.html#StructureGraph.get_subgraphs_as_molecules
    and removed the duplicate check
    Args:
        structure_graph ( pymatgen.analysis.graphs.StructureGraph): Structuregraph
        use_weights (bool): If True, use weights for the edge matching
        return_unique (bool): If true, it only returns the unique molecules.
            If False, it will return all molecules that are completely included in the unit cell
            and fragments of the ones that are only partly in the cell

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

    if return_unique:
        for subgraph in molecule_subgraphs:
            already_present = [
                nx.is_isomorphic(
                    subgraph, g, node_match=node_match, edge_match=edge_match
                )
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

    def relabel_graph(multigraph):
        mapping = dict(zip(multigraph, range(0, len(multigraph.nodes()))))
        return nx.readwrite.json_graph.adjacency_data(
            nx.relabel_nodes(multigraph, mapping)
        )

    if return_unique:
        mol, idx = make_mols(unique_subgraphs, center=True)
        return_subgraphs = unique_subgraphs
        return (
            mol,
            [
                MoleculeGraph(mol, relabel_graph(graph))
                for mol, graph in zip(mol, return_subgraphs)
            ],
            idx,
        )

    mol, idx = make_mols(molecule_subgraphs)
    return_subgraphs = [
        MoleculeGraph(mol, relabel_graph(graph))
        for mol, graph in zip(mol, molecule_subgraphs)
    ]

    mol, return_subgraphs, idx = _select_parts_in_cell(
        mol, return_subgraphs, idx, len(structure_graph)
    )

    return mol, return_subgraphs, idx
