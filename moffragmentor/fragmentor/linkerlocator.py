# -*- coding: utf-8 -*-
"""Based on the node location, locate the linkers"""
from typing import List, Tuple

import numpy as np
from pymatgen.core import Structure

from ..sbu import Linker, LinkerCollection
from ..utils import _flatten_list_of_sets
from .molfromgraph import get_subgraphs_as_molecules

__all__ = ["create_linker_collection", "identify_linker_binding_indices"]


def _pick_linker_indices(
    idxs: List[List[int]], centers, coordinates, all_node_branching_indices
) -> Tuple[List[int], List[int]]:
    """Trying to have a more reasonable way to filter out linkers
    (of multiple versions of the linker that might be wrapped across a unit cell)"""
    counter = 0
    unique_branching_site_centers = {}
    unique_branching_sites_indices = {}
    has_branch_point = []
    for idx, center, coords in zip(idxs, centers, coordinates):
        intersection = set(idx) & all_node_branching_indices
        if len(intersection) >= 2:
            has_branch_point.append(counter)
            intersection = tuple(sorted(tuple(intersection)))
            norm = np.linalg.norm(coords - center)
            if intersection in unique_branching_site_centers:
                if unique_branching_site_centers[intersection] > norm:
                    unique_branching_site_centers[intersection] = norm
                    unique_branching_sites_indices[intersection] = counter
            else:
                unique_branching_site_centers[intersection] = norm
                unique_branching_sites_indices[intersection] = counter
        counter += 1

    return unique_branching_sites_indices.values(), has_branch_point


def _get_connected_linkers(
    mof, branching_coordinates: List[np.array], linker_collection
) -> List[Tuple[int, np.array, np.array]]:
    """The insight of this function is that the branching
    indices outside the cell a node might
    be bound to are periodic images of the ones in the cell"""
    linked_to = []
    for branching_coordinate in branching_coordinates:

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
    branching_coordinates = []

    for node in node_collection:
        branching_coordinates.append(node.branching_coords)

    for i, node in enumerate(node_collection):
        edge_dict[i] = _get_connected_linkers(
            mof,
            branching_coordinates[i],
            linker_collection,
        )

    return edge_dict


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
        filter_in_cell=False,
        disable_boundary_crossing_check=True,
    )
    # Third: pick those molecules that are closest to the UC
    # ToDo: we should be able to skip this
    linker_indices, _ = _pick_linker_indices(
        idxs, centers, coordinates, node_location_result.branching_indices
    )
    # Fourth: collect all linkers in a linker collection
    for i, (mol, graph, idx, center) in enumerate(zip(mols, graphs, idxs, centers)):
        idxs = set(idx)
        branching_indices = node_location_result.branching_indices & idxs
        linker = Linker(
            molecule=mol,
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
        )

        if i in linker_indices:
            linkers.append(linker)

    linker_collection = LinkerCollection(linkers)

    # Fith: the edge dict describes the connections between metal clusters and linkers
    edge_dict = _get_edge_dict(mof, node_collection, linker_collection)
    return linker_collection, edge_dict


def create_linker_collection(
    mof,
    node_location_result,
    node_collection,
    unbound_solvents,
) -> Tuple[LinkerCollection, dict]:
    """Based on MOF, node locaion and unbound solvent location locate the linkers"""
    linker_collection, edge_dict = _create_linkers_from_node_location_result(
        mof, node_location_result, node_collection, unbound_solvents
    )
    return linker_collection, edge_dict


def identify_linker_binding_indices(mof, connecting_paths, indices):
    relevant_indices = connecting_paths & set(indices)

    # ToDo: this step is currently time limiting.
    # We should be able to skip it
    # my_new_graph = structure_graph.__copy__()
    # my_new_graph.structure = Structure.from_sites(my_new_graph.structure.sites)
    # my_new_graph.remove_nodes(
    #     [i for i in range(len(my_new_graph.structure)) if i not in relevant_indices]
    # )

    # _, mg, idx, _, _ = get_subgraphs_as_molecules(
    #     my_new_graph,
    #     filter_in_cell=True,
    #     return_unique=False,
    #     disable_boundary_crossing_check=False,
    # )

    # idx = {str(sorted(i)): mg for i, mg in zip(idx, mg)}

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
