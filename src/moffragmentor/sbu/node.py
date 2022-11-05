# -*- coding: utf-8 -*-
"""Create Python containers for node building blocks.

Here we understand metal clusters as nodes.
"""
from collections import defaultdict
from typing import Set

import numpy as np
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.core import Molecule, Structure
from structuregraph_helpers.create import get_nx_graph_from_edge_tuples
from structuregraph_helpers.subgraph import get_subgraphs_as_molecules

from moffragmentor.fragmentor.molfromgraph import wrap_molecule
from moffragmentor.utils import remove_all_nodes_not_in_indices

from .sbu import SBU
from ..fragmentor.splitter import _sites_and_classified_indices_from_indices
from ..utils import (
    _not_relevant_structure_indices,
    _reindex_list_of_tuple,
    get_neighbors_from_nx_graph,
)

__all__ = ["Node"]

_BINDING_DUMMY = "Xe"
_BRANCHING_DUMMY = "Kr"


def _extract_and_wrap(node_indices, all_branching_indices, all_binding_indices, mof):
    # first, create a molecule and molecule graph from those indices
    # that intersect with the node indices
    branching_in_node = [i for i in all_branching_indices if i in node_indices]
    binding_in_node = [i for i in all_binding_indices if i in node_indices]
    graph_ = mof.structure_graph.__copy__()
    remove_all_nodes_not_in_indices(
        graph_, list(node_indices) + branching_in_node + binding_in_node
    )
    _mols, graphs, idxs, centers, coordinates = get_subgraphs_as_molecules(
        graph_,
        return_unique=True,
        filter_in_cell=True,
        disable_boundary_crossing_check=False,
        prune_long_edges=True,
    )
    assert len(_mols) == 1
    mol, mapping = wrap_molecule(idxs[0], mof)

    # now, also create one molecule from the node indices, where we replace
    # binding and branching indices with dummy atoms
    node_metal_atoms = [i for i in node_indices if i in mof.metal_indices]
    binding_to_node_metal = set(sum([mof.get_neighbor_indices(i) for i in node_metal_atoms], []))
    binding_to_node_metal = binding_to_node_metal.intersection(all_binding_indices)
    branching_to_node_metal = set(sum([mof.get_neighbor_indices(i) for i in node_metal_atoms], []))
    binding_neighbors = sum([mof.get_neighbor_indices(i) for i in binding_to_node_metal], [])
    branching_to_binding = set(binding_neighbors).intersection(all_branching_indices)

    # Now, make a copy of the structure and replace the indices with dummy atoms
    dummy_structure = Structure.from_sites(mof.sites)
    for i in binding_to_node_metal:
        dummy_structure.replace(i, _BINDING_DUMMY)
    for i in branching_to_binding + branching_to_node_metal:
        dummy_structure.replace(i, _BRANCHING_DUMMY)

    to_delete = [
        i
        for i in range(len(dummy_structure))
        if i
        not in list(node_indices)
        + list(binding_to_node_metal)
        + list(branching_to_binding)
        + list(branching_to_node_metal)
    ]
    graph_w_dummy = mof.structure_graph.__copy__()
    graph_w_dummy.structure = dummy_structure
    graph_w_dummy.remove_nodes_from(to_delete)
    (
        _mols_w_dummy,
        graphs_w_dummy,
        idxs_w_dummy,
        centers_w_dummy,
        coordinates_with_dummy,
    ) = get_subgraphs_as_molecules(
        graph_w_dummy,
        return_unique=True,
        filter_in_cell=True,
        disable_boundary_crossing_check=False,
        prune_long_edges=True,
    )
    assert len(_mols_w_dummy) == 1
    mol_w_dummy, mapping_w_dummy = wrap_molecule(idxs_w_dummy[0], mof)


def _find_node_hidden_indices(node_indices, all_binding_indices, all_branching_indices, mof):
    # find the binding indices bound to node
    node_neighbors = sum([mof.get_neighbor_indices(i) for i in node_indices], [])
    relevant_binding_indices = [i for i in node_neighbors if i in all_binding_indices]
    relevant_binding_indices_neighbors = sum(
        [mof.get_neighbor_indices(i) for i in relevant_binding_indices], []
    )
    relevant_branching_indices = [
        i for i in relevant_binding_indices_neighbors if i in all_branching_indices
    ] + [i for i in all_branching_indices if i in node_neighbors]
    hidden_indices = relevant_branching_indices + relevant_binding_indices
    return set(hidden_indices)


def node_from_mof_and_indices(  # pylint:disable=too-many-locals, too-many-arguments
    cls, mof, node_indices, all_branching_indices, all_binding_indices, all_connecting_paths
):
    """Create a node from a MOF and a list of indices of different types."""
    branching_indices = node_indices & all_branching_indices
    binding_indices = node_indices & all_binding_indices
    connecting_paths = all_connecting_paths & node_indices

    hidden_indices = _find_node_hidden_indices(
        node_indices, all_binding_indices, all_branching_indices, mof
    )

    graph_ = mof.structure_graph.__copy__()
    graph_.structure = Structure.from_sites(graph_.structure.sites)
    to_delete = _not_relevant_structure_indices(mof.structure, node_indices)
    graph_.remove_nodes(to_delete)

    sites_and_indices = _sites_and_classified_indices_from_indices(mof, node_indices)
    relevant_indices = node_indices - set(sites_and_indices.hidden_vertices)

    selected_positions, selected_species, selected_edge_indices = [], [], []

    # give us a quick and easy way to filter the edges we want to keep
    # for this reason we will use both edge "partners" as keys
    # and have a list of indices in the original edges list as the value
    edge_dict = defaultdict(list)
    for i, egde in enumerate(sites_and_indices.edges):
        edge_partner_1, edge_partner_2 = egde
        edge_dict[edge_partner_1].append(i)
        edge_dict[edge_partner_2].append(i)

    for i, index in enumerate(node_indices):
        if index in relevant_indices:
            selected_positions.append(sites_and_indices.cartesian_coordinates[i])
            selected_species.append(sites_and_indices.species[i])
            edge_idxs = edge_dict[index]
            for edge_idx in edge_idxs:
                edge = sites_and_indices.edges[edge_idx]
                if (edge[0] in relevant_indices) and (edge[1] in relevant_indices):
                    selected_edge_indices.append(edge_idx)

    index_map = dict(zip(relevant_indices, range(len(relevant_indices))))
    molecule = Molecule(selected_species, selected_positions)
    selected_edges = [sites_and_indices.edges[i] for i in selected_edge_indices]
    selected_edges = _reindex_list_of_tuple(selected_edges, index_map)
    edge_dict = dict(zip(selected_edges, [None] * len(selected_edges)))
    molecule_graph = MoleculeGraph.with_edges(molecule, edge_dict)
    center = np.mean(molecule.cart_coords, axis=0)
    graph_branching_indices = branching_indices & node_indices

    idx = [i for i in relevant_indices if i in node_indices]

    # Add the branching indices here?
    mol_w_hidden, mapping = wrap_molecule(
        idx + sites_and_indices.hidden_vertices + list(hidden_indices), mof
    )

    # perhaps than delete here the hidden ones again to to get the mol without the hidden ones

    node = cls(
        molecule=mol_w_hidden,
        molecule_graph=molecule_graph,
        center=center,
        graph_branching_indices=graph_branching_indices,
        binding_indices=binding_indices,
        original_indices=idx + sites_and_indices.hidden_vertices,
        persistent_non_metal_bridged=sites_and_indices.persistent_non_metal_bridged_components,
        terminal_in_mol_not_terminal_in_struct=sites_and_indices.hidden_vertices,
        connecting_paths=connecting_paths,
        molecule_original_indices_mapping=mapping,
    )

    return node


class Node(SBU):
    """Container for metal cluster building blocks.

    Will typically automatically be constructured by the fragmentor.
    """

    @classmethod
    def from_mof_and_indices(  # pylint:disable=too-many-arguments
        cls,
        mof,
        node_indices: Set[int],
        branching_indices: Set[int],
        binding_indices: Set[int],
        connecting_paths: Set[int],
    ):
        """Build a node object from a MOF and some intermediate outputs of the fragmentation.

        Args:
            mof (MOF): The MOF to build the node from.
            node_indices (Set[int]): The indices of the nodes in the MOF.
            branching_indices (Set[int]): The indices of the branching points in the MOF
                that belong to this node.
            binding_indices (Set[int]): The indices of the binding points in the MOF
                that belong to this node.
            connecting_paths (Set[int]): The indices of the connecting paths in the MOF
                that belong to this node.

        Returns:
            A node object.
        """
        return node_from_mof_and_indices(
            cls,
            mof,
            node_indices,
            all_branching_indices=branching_indices,
            all_binding_indices=binding_indices,
            all_connecting_paths=connecting_paths,
        )

    def __repr__(self) -> str:
        """Return a string representation of the node."""
        return f"Node ({self.composition})"
