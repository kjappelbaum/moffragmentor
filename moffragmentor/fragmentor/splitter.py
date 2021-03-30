# -*- coding: utf-8 -*-
"""This module focusses on the extraction of pymatgen Molecules from a structure for which we know the branching points / node/linker indices"""
from collections import defaultdict
from copy import deepcopy
from typing import List, Tuple

import networkx as nx
import numpy as np
from pymatgen import Molecule
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph

from ..molecule import NonSbuMolecule, NonSbuMoleculeCollection
from ..utils.periodic_graph import _get_reverse_supergraph_index_map

__all__ = ["get_subgraphs_as_molecules"]


def _connected_component_subgraph(graph, component):
    # Doing it this way to keep the indices
    g_ = deepcopy(graph)
    g_.remove_nodes_from([n for n in g_ if n not in set(component)])
    return g_


def _select_parts_in_cell(
    molecules: List[Molecule],
    graphs: List[MoleculeGraph],
    indices: List[List[int]],
    indices_here: List[List[int]],
    molecule_size: int = np.inf,
) -> Tuple[List[Molecule], List[MoleculeGraph], List[List[int]]]:

    valid_indices = defaultdict(list)

    for i, indices_here_l in enumerate(indices_here):
        sorted_idx = sorted(indices[i])
        valid_indices[str(sorted_idx)].append(i)

    molecules_ = []
    selected_indices = []
    graphs_ = []

    for _, v in valid_indices.items():
        selected_indices.append(indices[v[0]])
        molecules_.append(molecules[v[0]])
        graphs_.append(graphs[v[0]])

    return molecules_, graphs_, selected_indices


def get_subgraphs_as_molecules(
    structure_graph: StructureGraph,
    use_weights: bool = False,
    return_unique: bool = True,
    original_len: int = None,
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
        supercell_sg.graph.subgraph(c).copy()
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
        indices_here = []
        for subgraph in molecule_subgraphs:
            coords = [supercell_sg.structure[n].coords for n in subgraph.nodes()]
            species = [supercell_sg.structure[n].specie for n in subgraph.nodes()]
            binding = [
                supercell_sg.structure[n].properties["binding"]
                for n in subgraph.nodes()
            ]
            idx = [subgraph.nodes[n]["idx"] for n in subgraph.nodes()]
            idx_here = [n for n in subgraph.nodes()]
            molecule = Molecule(species, coords, site_properties={"binding": binding})

            # shift so origin is at center of mass
            if center:
                molecule = molecule.get_centered_molecule()
            indices.append(idx)
            molecules.append(molecule)
            indices_here.append(idx_here)
        return molecules, indices, indices_here

    def relabel_graph(multigraph):
        mapping = dict(zip(multigraph, range(0, len(multigraph.nodes()))))
        return nx.readwrite.json_graph.adjacency_data(
            nx.relabel_nodes(multigraph, mapping)
        )

    if return_unique:
        mol, idx, indices_here = make_mols(unique_subgraphs, center=True)
        return_subgraphs = unique_subgraphs
        return (
            mol,
            [
                MoleculeGraph(mol, relabel_graph(graph))
                for mol, graph in zip(mol, return_subgraphs)
            ],
            idx,
        )

    mol, idx, indices_here = make_mols(molecule_subgraphs)

    return_subgraphs = [
        MoleculeGraph(mol, relabel_graph(graph))
        for mol, graph in zip(mol, molecule_subgraphs)
    ]

    if original_len is not None:
        len_limit = original_len
    else:
        len_limit = len(structure_graph)
    mol, return_subgraphs, idx = _select_parts_in_cell(
        mol, return_subgraphs, idx, indices_here, len_limit
    )

    return mol, return_subgraphs, idx


def get_floating_solvent_molecules(mof) -> NonSbuMoleculeCollection:
    """Create a collection of NonSbuMolecules
    from a MOF

    Args:
        mof (MOF): instance of MOF

    Returns:
        NonSbuMoleculeCollection: collection of NonSbuMolecules
    """
    _, _, idx = get_subgraphs_as_molecules(mof.structure_graph, return_unique=False)
    molecules = []
    # Todo: this mapping is probably something that should go into the get_subgraphs_as_molecules function
    mapping = _get_reverse_supergraph_index_map(len(mof))
    for id in idx:
        new_ids = [mapping[i] for i in id]
        molecules.append(
            NonSbuMolecule.from_structure_graph_and_indices(
                mof.structure_graph * (3, 3, 3), new_ids
            )
        )

    return NonSbuMoleculeCollection(molecules)
