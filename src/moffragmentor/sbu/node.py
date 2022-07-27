# -*- coding: utf-8 -*-
"""Create Python containers for node building blocks.

Here we understand metal clusters as nodes.
"""
from collections import defaultdict
from typing import Set

import numpy as np
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.core import Molecule, Structure

from moffragmentor.fragmentor.molfromgraph import wrap_molecule

from .sbu import SBU
from ..fragmentor.splitter import _sites_and_classified_indices_from_indices
from ..utils import (
    _not_relevant_structure_indices,
    _reindex_list_of_tuple,
    get_neighbors_from_nx_graph,
    get_nx_graph_from_edge_tuples,
)

__all__ = ["Node"]


def node_from_mof_and_indices(  # pylint:disable=too-many-locals, too-many-arguments
    cls, mof, node_indices, branching_indices, binding_indices, connecting_paths
):
    """Create a node from a MOF and a list of indices of different types."""
    node_indices = node_indices | connecting_paths  # This should actually not be necessary
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
    closest_branching_index_in_molecule = []

    graph = get_nx_graph_from_edge_tuples(sites_and_indices.edges)
    for branching_index in branching_indices:
        if branching_index not in relevant_indices:
            # now we need to find the closest neighbor in the
            # set of vertices that are in the molecule
            # ToDo: this is currently an assumption that it is terminal and the
            # next partner then already is in the molecule,
            # we could recursively call or get all the paths and then get the shortest
            new_candidates = get_neighbors_from_nx_graph(graph, branching_index)[0]
            closest_branching_index_in_molecule.append(new_candidates)
        else:
            closest_branching_index_in_molecule.append(branching_index)
    idx = [i for i in relevant_indices if i in node_indices]
    mol_w_hidden, mapping = wrap_molecule(idx + sites_and_indices.hidden_vertices, mof)
    node = cls(
        molecule=mol_w_hidden,
        molecule_graph=molecule_graph,
        center=center,
        graph_branching_indices=graph_branching_indices,
        closest_branching_index_in_molecule=closest_branching_index_in_molecule,
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
            branching_indices,
            binding_indices,
            connecting_paths=connecting_paths,
        )
