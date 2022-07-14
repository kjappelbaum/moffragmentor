# -*- coding: utf-8 -*-
"""Based on the node location, locate the linkers."""
from typing import Iterable, List, Optional, Tuple

import numpy as np
from loguru import logger
from pymatgen.core import Lattice, Structure

from .molfromgraph import _is_any_atom_in_cell, get_subgraphs_as_molecules, wrap_molecule
from ..sbu import Linker, LinkerCollection
from ..utils import _flatten_list_of_sets

__all__ = ["create_linker_collection", "identify_linker_binding_indices"]


def _pick_linker_indices(
    idxs: List[List[int]],
    centers: Iterable[np.array],
    coordinates: Iterable[np.array],
    all_node_branching_indices: Iterable[int],
    two_branching_indices: bool = True,
    lattice: Optional[Lattice] = None,
) -> Tuple[List[int], List[int]]:
    """Pick the relevant linkers.

    Trying to have a more reasonable way to filter out linkers
    (of multiple versions of the linker that might be wrapped across a unit cell).

    Args:
        idxs (List[List[int]]): List of list of linker indices
        centers (Iterable[np.array]): List of linker centers
        coordinates (Iterable[np.array]): List of linker coordinates
        all_node_branching_indices (Iterable[int]): List of all node branching indices
        two_branching_indices (bool): If True, only linkers with
            at least two branching indices are considers.
            Defaults to True.
        lattice (Optional[Lattice]): Lattice of the MOF.

    Returns:
        Tuple[List[int], List[int]]: List of linker indices and list of
            linker indices that have branching points
    """
    # ToDo: reusue some of this computation when we get the net.
    # We do a similar loop there
    threshold = 2 if two_branching_indices else 1
    counter = 0

    unique_branching_sites_indices = {}
    has_branch_point = []
    for idx, coords in zip(idxs, coordinates):
        intersection = set(idx) & all_node_branching_indices
        if intersection:
            branching_coords = lattice.get_fractional_coords(
                [coord for i, coord in enumerate(coords) if idx[i] in intersection]
            )
            if len(intersection) >= threshold:
                if _is_any_atom_in_cell(branching_coords):
                    has_branch_point.append(counter)
                    intersection = tuple(sorted(tuple(intersection)))
                    unique_branching_sites_indices[intersection] = counter
            counter += 1

    return unique_branching_sites_indices.values(), has_branch_point


def _create_linkers_from_node_location_result(
    mof, node_location_result, node_collection, unbound_solvent
) -> Tuple[LinkerCollection, dict]:
    linkers = []

    # First step: remove everything that is node or unbound solvent
    # bound solvent is part of the node by default
    all_node_indices = set()
    all_persistent_non_metal_bridges = set()
    for node_indices in node_location_result.nodes:
        all_node_indices.update(node_indices)

    for node in node_collection:
        all_persistent_non_metal_bridges.update(
            _flatten_list_of_sets(
                node._persistent_non_metal_bridged  # pylint:disable=protected-access
            )
        )

    not_linker_indices = (
        (
            all_node_indices
            - node_location_result.connecting_paths
            - node_location_result.branching_indices
            - all_persistent_non_metal_bridges
        )
        | set(unbound_solvent.indices)
        | set(mof.metal_indices) & all_node_indices
        # some metals might also be in the linker, e.g., in porphyrins
    )

    graph_ = mof.structure_graph.__copy__()
    graph_.structure = Structure.from_sites(graph_.structure.sites)
    graph_.remove_nodes(not_linker_indices)

    # Second step: extract the connected components
    # return all as molecules
    mols, graphs, idxs, centers, coordinates = get_subgraphs_as_molecules(
        graph_,
        return_unique=False,
        filter_in_cell=True,
        disable_boundary_crossing_check=True,
    )

    # Third: pick those molecules that are closest to the UC
    # ToDo: we should be able to skip this
    linker_indices, _ = _pick_linker_indices(
        idxs, centers, coordinates, node_location_result.branching_indices, lattice=mof.lattice
    )
    if len(linker_indices) == 0:
        logger.warning(
            "No linkers with two branching sites in molecule found. \
                 Looking for molecules with one branching site."
        )
        linker_indices, _ = _pick_linker_indices(
            idxs,
            centers,
            coordinates,
            node_location_result.branching_indices,
            two_branching_indices=False,
            lattice=mof.lattice,
        )

    # Fourth: collect all linkers in a linker collection
    for i, (mol, graph, idx, center, coord) in enumerate(
        zip(mols, graphs, idxs, centers, coordinates)
    ):
        idxs = set(idx)
        branching_indices = node_location_result.branching_indices & idxs
        linker = Linker(
            molecule=wrap_molecule(list(idxs), mof),
            molecule_graph=graph,
            center=center,
            graph_branching_indices=branching_indices,
            closest_branching_index_in_molecule=branching_indices,
            binding_indices=identify_linker_binding_indices(
                mof,
                node_location_result.connecting_paths,
                idx,
            ),
            original_indices=idx,
            connecting_paths=[],
            coordinates=coord,
            lattice=mof.lattice,
        )

        if i in linker_indices:
            linkers.append(linker)

    linker_collection = LinkerCollection(linkers)
    return linker_collection


def create_linker_collection(
    mof,
    node_location_result,
    node_collection,
    unbound_solvents,
) -> Tuple[LinkerCollection, dict]:
    """Based on MOF, node locaion and unbound solvent location locate the linkers"""
    linker_collection = _create_linkers_from_node_location_result(
        mof, node_location_result, node_collection, unbound_solvents
    )
    return linker_collection


def identify_linker_binding_indices(mof, connecting_paths, indices):
    relevant_indices = connecting_paths & set(indices)

    # ToDo: this step is currently time limiting.
    # We should be able to skip it
    # my_new_graph = structure_graph.__copy__()
    # my_new_graph.structure = Structure.from_sites(my_new_graph.structure.sites)
    # my_new_graph.remove_nodes(
    #     [i for i in range(len(my_new_graph.structure)) if i not in relevant_indices]
    # )

    # Now, we need to filter these index sets.
    # If they are of length 1 there is nothing we need to do
    # If they are longer, however, we need to do some reasoning
    filtered_indices = []

    metal_indices = set(mof.metal_indices)
    for idx in relevant_indices:
        neighbors = mof.get_neighbor_indices(idx)
        if not metal_indices.isdisjoint(set(neighbors)):
            filtered_indices.append(idx)

    return filtered_indices
