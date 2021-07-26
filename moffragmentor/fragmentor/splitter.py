# -*- coding: utf-8 -*-
"""This module focusses on the extraction of pymatgen Molecules from a structure for which we know the branching points / node/linker indices"""
from collections import defaultdict, namedtuple
from copy import deepcopy
from typing import List, Tuple

import networkx as nx
import numpy as np
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.core import Molecule

from ..molecule import NonSbuMolecule, NonSbuMoleculeCollection
from ..utils import (
    _get_cartesian_coords,
    _get_molecule_edge_label,
    _get_vertices_of_smaller_component_upon_edge_break,
    _is_any_atom_in_cell,
    _metal_in_edge,
    _reindex_list_of_tuple,
    revert_dict,
)

__all__ = ["get_subgraphs_as_molecules"]


def _select_parts_in_cell(
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


def get_subgraphs_as_molecules(
    structure_graph: StructureGraph,
    use_weights: bool = False,
    return_unique: bool = True,
    disable_boundary_crossing_check: bool = False,
    filter_in_cell: bool = True,
) -> Tuple[List[Molecule], List[MoleculeGraph], List[List[int]], List[np.ndarray]]:
    """Copied from
    http://pymatgen.org/_modules/pymatgen/analysis/graphs.html#StructureGraph.get_subgraphs_as_molecules
    and removed the duplicate check
    Args:
        structure_graph ( pymatgen.analysis.graphs.StructureGraph): Structuregraph
        use_weights (bool): If True, use weights for the edge matching
        return_unique (bool): If true, it only returns the unique molecules.
            If False, it will return all molecules that are completely included in the unit cell
            and fragments of the ones that are only partly in the cell
        filter_in_cell (bool): If True, it will only return molecules that have at least one atom
            in the cell

    Returns:
        Tuple[List[Molecule], List[MoleculeGraph], List[List[int]], List[np.ndarray]]
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
        mol_centers = []
        coordinates = []
        for subgraph in molecule_subgraphs:
            coords = [supercell_sg.structure[n].coords for n in subgraph.nodes()]
            species = [supercell_sg.structure[n].specie for n in subgraph.nodes()]
            # binding = [
            #     supercell_sg.structure[n].properties["binding"]
            #     for n in subgraph.nodes()
            # ]
            idx = [subgraph.nodes[n]["idx"] for n in subgraph.nodes()]
            idx_here = [n for n in subgraph.nodes()]
            molecule = Molecule(
                species, coords
            )  #  site_properties={"binding": binding}
            mol_centers.append(
                np.mean(supercell_sg.structure.cart_coords[idx_here], axis=0)
            )
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
        return nx.readwrite.json_graph.adjacency_data(
            nx.relabel_nodes(multigraph, mapping)
        )

    if return_unique:
        mol, idx, indices_here, centers, coordinates = make_mols(
            unique_subgraphs, center=True
        )
        return_subgraphs = unique_subgraphs
        return (
            mol,
            [
                MoleculeGraph(mol, relabel_graph(graph))
                for mol, graph in zip(mol, return_subgraphs)
            ],
            idx,
            centers,
            coordinates,
        )

    mol, idx, indices_here, centers, coordinates = make_mols(molecule_subgraphs)

    return_subgraphs = [
        MoleculeGraph(mol, relabel_graph(graph))
        for mol, graph in zip(mol, molecule_subgraphs)
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


def get_floating_solvent_molecules(mof) -> NonSbuMoleculeCollection:
    """Create a collection of NonSbuMolecules
    from a MOF

    Args:
        mof (MOF): instance of MOF

    Returns:
        NonSbuMoleculeCollection: collection of NonSbuMolecules
    """
    mols, graphs, idx, _, _ = get_subgraphs_as_molecules(
        mof.structure_graph, return_unique=False
    )
    molecules = []

    for mol, graph, id in zip(mols, graphs, idx):
        molecules.append(NonSbuMolecule(mol, graph, id))

    return NonSbuMoleculeCollection(molecules)


_SitesAndIndicesOutput = namedtuple(
    "SitesAndIndicesOutput",
    [
        "cartesian_coordinates",
        "species",
        "edges",
        "index_mapping",
        "hidden_vertices",
        "persistent_non_metal_bridged_components",
    ],
)


def _sites_and_classified_indices_from_indices(
    mof, indices: set
) -> Tuple[Molecule, List[int], List[int]]:
    """
    Given the indices that we identified to belong to the group
    1) Build a molecule from the structure, handling the periodic images
    2) Build the molecule graph from the structure graph, handling the periodic images
    3) Flag vertices in the graph that we call "hidden". Those have only one neighbor
        in the molecule but multiple in the structure.
    This happens, for example, in MOF-74 where the "connecting site" is not on the carboxy carbon
        given that one of the carboxy Os is not connected to the metal cluster
    To get reasonable nodes/linkers we will remove these from the node.
    We also flag indices that are connected via a bridge that is also a bridge in the original graph.
        This is, for example, the case for one carboxy O in MOF-74
    """
    new_positions = []
    persistent_non_metal_bridged_components = []
    hidden_vertices = []
    # we store edges as list of tuples as we can
    # do a cleaner loop for reindexing
    edges = []
    index_mapping = {}

    # cast set to list, so we can loop
    indices = list(indices)
    species = []
    for new_idx, idx in enumerate(indices):
        new_positions.append(_get_cartesian_coords(mof, indices[0], idx))
        index_mapping[idx] = new_idx
        # This will fail for alloys/fractional occupancy
        # ToDo: check early if this might be an issue
        species.append(str(mof.structure[idx].specie))
        neighbors = mof.structure_graph.get_connected_sites(idx)
        # counter for the number of neighbors that are also part
        # of the molecule, i.e., in `indices`
        num_neighbors_in_molecule = 0
        for neighbor in neighbors:
            # we also identified the neighbor to be part of the molecule
            if neighbor.index in indices:
                edges.append(_get_molecule_edge_label(idx, neighbor.index))
                num_neighbors_in_molecule += 1
        if (num_neighbors_in_molecule == 1) and len(neighbors) > 1:
            hidden_vertices.append(idx)

    reindexed_edges = _reindex_list_of_tuple(edges, index_mapping)
    edge_dict = dict(zip(reindexed_edges, [None] * len(reindexed_edges)))

    mol = Molecule(species, new_positions)
    graph = MoleculeGraph.with_edges(mol, edge_dict)

    new_bridges = set(nx.bridges(nx.Graph(graph.graph.to_undirected())))
    new_index_to_old_index = revert_dict(index_mapping)
    new_bridges_w_old_indices = _reindex_list_of_tuple(
        new_bridges, new_index_to_old_index
    )

    relevant_bridges = []

    for new_index_bridge, old_index_bridge in zip(
        new_bridges, new_bridges_w_old_indices
    ):
        if not _metal_in_edge(mol, new_index_bridge):
            if mof._leads_to_terminal(  # pylint:disable=protected-access
                old_index_bridge
            ):
                relevant_bridges.append(new_index_bridge)

    for bridge in relevant_bridges:
        persistent_non_metal_bridged_components.append(
            _get_vertices_of_smaller_component_upon_edge_break(
                graph.graph.to_undirected(), bridge
            )
        )

    persistent_non_metal_bridged_components_old_idx = []
    for subset in persistent_non_metal_bridged_components:
        indices = [new_index_to_old_index[i] for i in subset]
        persistent_non_metal_bridged_components_old_idx.append(indices)

    return _SitesAndIndicesOutput(
        new_positions,
        species,
        edges,
        index_mapping,
        hidden_vertices,
        persistent_non_metal_bridged_components_old_idx,
    )
