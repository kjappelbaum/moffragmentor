# -*- coding: utf-8 -*-
"""Based on the node location, locate the linkers"""
from typing import Tuple

import numpy as np
from pymatgen.core import Structure
from structuregraph_helpers.subgraph import get_subgraphs_as_molecules

from moffragmentor.fragmentor.molfromgraph import wrap_molecule
from moffragmentor.sbu import Linker, LinkerCollection

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


def _create_linkers_from_node_location_result(  # pylint:disable=too-many-locals
    mof, node_location_result, node_collection, unbound_solvent, bound_solvent
) -> Tuple[LinkerCollection, dict]:
    linkers = []

    # First step: remove everything that is node or unbound solvent
    # bound solvent is part of the node by default
    all_node_indices = set()
    # all_persistent_non_metal_bridges = set()
    for node_indices in node_location_result.nodes:
        all_node_indices.update(node_indices)

    # for node in node_collection:
    #     all_persistent_non_metal_bridges.update(
    #         _flatten_list_of_sets(
    #             node._persistent_non_metal_bridged  # pylint:disable=protected-access
    #         )
    #     )

    not_linker_indices = (
        (
            all_node_indices
            & (
                node_location_result.connecting_paths
                - node_location_result.branching_indices
                - node_location_result.binding_indices
                - set(sum(node_location_result.to_terminal_from_branching.values(), []))
            )
        )
        | set(unbound_solvent.indices)
        | set(bound_solvent.indices)
        | set(mof.metal_indices) & all_node_indices
        # some metals might also be in the linker, e.g., in porphyrins
    )
    graph_ = mof.structure_graph.__copy__()
    graph_.structure = Structure.from_sites(graph_.structure.sites)
    graph_.remove_nodes(not_linker_indices)

    # Second step: extract the connected components
    # return all as molecules
    _mols, graphs, idxs, _centers, coordinates = get_subgraphs_as_molecules(
        graph_,
        return_unique=False,
        filter_in_cell=False,
        disable_boundary_crossing_check=True,
        prune_long_edges=True,
    )

    # Third: pick those molecules that are closest to the UC
    # ToDo: we should be able to skip this
    linker_indices, _coords = _pick_central_linker_indices(mof, coordinates)
    found_frac_centers = set()
    found_hashes = set()
    for linker_index in linker_indices:
        idx = idxs[linker_index]
        branching_indices = node_location_result.branching_indices & set(idx)
        mol, mapping = wrap_molecule(idx, mof)
        linker = Linker(
            molecule=mol,
            molecule_graph=graphs[linker_index],
            graph_branching_indices=branching_indices,
            binding_indices=identify_linker_binding_indices(
                mof,
                node_location_result.connecting_paths,
                idx,
            ),
            molecule_original_indices_mapping=mapping,
        )
        frac_center = mof.structure.lattice.get_fractional_coords(mol.center_of_mass)
        frac_center -= np.floor(frac_center)
        frac_center = (
            np.round(frac_center[0], 4),
            np.round(frac_center[1], 4),
            np.round(frac_center[2], 4),
        )
        if linker.hash not in found_hashes and frac_center not in found_frac_centers:
            found_hashes.add(linker.hash)
            linkers.append(linker)
            found_frac_centers.add(frac_center)
    linker_collection = LinkerCollection(linkers)

    return linker_collection


def create_linker_collection(
    mof,
    node_location_result,
    node_collection,
    unbound_solvents,
    bound_solvents,
) -> Tuple[LinkerCollection, dict]:
    """Based on MOF, node locaion and unbound solvent location locate the linkers"""
    linker_collection = _create_linkers_from_node_location_result(
        mof, node_location_result, node_collection, unbound_solvents, bound_solvents
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


def _check_linker(linker, mof):
    # check that no linker has more than two metals
    metal_indices = set(mof.metal_indices)
    return len(metal_indices & set(linker.original_indices)) < 2


def check_linkers(linker_collection, mof):
    """Check if the linkers are valid"""
    return all(_check_linker(linker, mof) for linker in linker_collection)
