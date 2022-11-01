# -*- coding: utf-8 -*-
"""Funnctions that can be used to locate bound and unbound sovlent"""
from collections import OrderedDict
from typing import List, Set

from structuregraph_helpers.subgraph import get_subgraphs_as_molecules

from ._graphsearch import _has_path_to_any_other_metal, recursive_dfs_until_terminal
from .filter import bridges_across_cell
from ..molecule import NonSbuMolecule, NonSbuMoleculeCollection
from ..utils import _get_metal_sublist

__all__ = ["get_floating_solvent_molecules", "get_all_bound_solvent_molecules"]


def get_floating_solvent_molecules(mof) -> NonSbuMoleculeCollection:
    """Create a collection of NonSbuMolecules from a MOF.

    Args:
        mof (MOF): instance of MOF

    Returns:
        NonSbuMoleculeCollection: collection of NonSbuMolecules
    """
    mols, graphs, idx, _, _ = get_subgraphs_as_molecules(mof.structure_graph, return_unique=False)
    molecules = []

    for mol, graph, identifier in zip(mols, graphs, idx):
        molecules.append(NonSbuMolecule(mol, graph, identifier))

    return NonSbuMoleculeCollection(molecules)


def _get_solvent_molecules_bound_to_node(mof, node_atoms: Set[int]) -> NonSbuMoleculeCollection:
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
        # check if the solvent candidate has bridge to two different metals:
        # if so, it is not a solvent
        # that is, we consider capping groups different from bound solvent
        neighbors: Set[int] = set()
        for ind in solvent_ind:
            # this is here actually not enough.
            # we also need to look at the images
            neighbors.update(mof.get_neighbor_indices(ind))

        if not len(neighbors.intersection(mof.metal_indices)) > 1:
            molecules.append(
                NonSbuMolecule.from_structure_graph_and_indices(mof.structure_graph, solvent_ind)
            )

    return NonSbuMoleculeCollection(molecules)


def get_all_bound_solvent_molecules(
    mof, node_atom_sets: List[Set[int]]
) -> NonSbuMoleculeCollection:
    """Identify all bound solvent molecules.

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
        non_sbu_molecule_collections += _get_solvent_molecules_bound_to_node(mof, node_atom_set)

    return non_sbu_molecule_collections


def find_solvent_molecule_indices(mof, index: int, starting_metal: int) -> List[int]:
    """Find all the indices that belong to a solvent molecule.

    Args:
        mof (MOF): index
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
            # Note: we cannot do a simple bridge check here because we would not find things such as acetate
            if not _has_path_to_any_other_metal(mof, metal_neighbor, metal_index):
                potential_solvent_indices = find_solvent_molecule_indices(
                    mof, metal_neighbor, metal_index
                )
                if not bridges_across_cell(mof, potential_solvent_indices):
                    solvent_connections.add(metal_neighbor)
                    solvent_indices.append(potential_solvent_indices)

            else:
                good_connections.add(metal_neighbor)

    return OrderedDict(
        [
            ("solvent_indices", solvent_indices),
            ("solvent_connections", solvent_connections),
            ("non_solvent_connections", good_connections),
        ]
    )
