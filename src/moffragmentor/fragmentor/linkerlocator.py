# -*- coding: utf-8 -*-
"""Based on the node location, locate the linkers"""
from typing import List, Tuple

import numpy as np
from loguru import logger
from pymatgen.core import Lattice, Structure

from moffragmentor.net import in_cell

from .molfromgraph import get_subgraphs_as_molecules, wrap_molecule
from ..sbu import Linker, LinkerCollection
from ..utils import _flatten_list_of_sets

__all__ = ["create_linker_collection", "identify_linker_binding_indices"]


def _is_site_in_cell(coord):
    return np.all((coord >= 1) & (coord <= 2))


def _any_site_in_cell(coords):
    return np.any([_is_site_in_cell(coord) for coord in coords])


def _pick_central_linker_indices(mof, coords):
    selected = []
    coordinates = []
    for i, coord in enumerate(coords):
        # we mimic that the central (1,1,1)-(2,2,2) unit cell is the
        # original unit cell
        frac = mof.structure.lattice.get_fractional_coords(coord)
        if _any_site_in_cell(frac):
            # the (1,1,1) coordinate is the origin of the original unit cell
            trans_frac = frac - np.array([1, 1, 1])
            old_cart = mof.structure.lattice.get_cartesian_coords(trans_frac)
            selected.append(i)
            coordinates.append(old_cart)
    return selected, coordinates


def number_branching_points_in_cell(coordinates, lattice):
    in_cell = 0
    for coord in coordinates:
        frac_coord = lattice.get_fractional_coords(coord)
        if np.all(frac_coord <= 1) & np.all(frac_coord >= 0):
            in_cell += 1

    return in_cell


def _pick_linker_indices(
    idxs: List[List[int]],
    centers,
    coordinates,
    all_node_branching_indices,
    lattice: Lattice,
    two_branching_indices=True,
) -> Tuple[List[int], List[int]]:
    """Trying to have a more reasonable way to filter out linkers
    (of multiple versions of the linker that might be wrapped across a unit cell)"""
    threshold = 2 if two_branching_indices else 1
    counter = 0
    unique_branching_site_centers = {}
    unique_branching_sites_indices = {}
    has_branch_point = []
    for idx, center, coords in zip(idxs, centers, coordinates):
        intersection = set(idx) & all_node_branching_indices
        if len(intersection) >= threshold:
            has_branch_point.append(counter)
            intersection = tuple(sorted(tuple(intersection)))
            norm = number_branching_points_in_cell(coords, lattice)
            if not (np.abs(lattice.get_fractional_coords(coords)) >= 2).any():
                if intersection in unique_branching_site_centers:
                    if unique_branching_site_centers[intersection] < norm:
                        unique_branching_site_centers[intersection] = norm
                        unique_branching_sites_indices[intersection] = counter
                else:
                    unique_branching_site_centers[intersection] = norm
                    unique_branching_sites_indices[intersection] = counter
        counter += 1

    return unique_branching_sites_indices.values(), has_branch_point


def _create_linkers_from_node_location_result(  # pylint:disable=too-many-locals
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
            & (
                node_location_result.connecting_paths
                - node_location_result.branching_indices
                - node_location_result.binding_indices
            )
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
        filter_in_cell=False,
        disable_boundary_crossing_check=True,
        prune_long_edges=True,
    )

    # Third: pick those molecules that are closest to the UC
    # ToDo: we should be able to skip this
    linker_indices, coords = _pick_central_linker_indices(mof, coordinates)

    found_hashes = set()
    for i, linker_index in enumerate(linker_indices):
        idx = idxs[linker_index]
        center = centers[linker_index]
        coords_ = coords[i]
        branching_indices = node_location_result.branching_indices & set(idx)
        linker = Linker(
            molecule=wrap_molecule(idx, mof),
            molecule_graph=graphs[linker_index],
            center=center,
            graph_branching_indices=branching_indices,
            closest_branching_index_in_molecule=branching_indices,
            binding_indices=identify_linker_binding_indices(
                mof,
                node_location_result.connecting_paths,
                idx,
            ),
            coordinates=coords_,
            original_indices=idx,
            connecting_paths=[],
        )
        if linker.hash not in found_hashes:
            assert len(idx) == len(coords_)
            found_hashes.add(linker.hash)
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

    filtered_indices = []

    metal_indices = set(mof.metal_indices)
    for idx in relevant_indices:
        neighbors = mof.get_neighbor_indices(idx)
        if not metal_indices.isdisjoint(set(neighbors)):
            filtered_indices.append(idx)

    return filtered_indices
