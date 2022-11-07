# -*- coding: utf-8 -*-
"""Create Python containers for node building blocks.

Here we understand metal clusters as nodes.
"""
from collections import defaultdict
from typing import Set

from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.core import Molecule, Site, Structure

from moffragmentor.fragmentor.molfromgraph import wrap_molecule
from moffragmentor.fragmentor.splitter import _sites_and_classified_indices_from_indices
from moffragmentor.sbu.sbu import SBU
from moffragmentor.utils import _reindex_list_of_tuple

__all__ = ["Node"]

_BINDING_DUMMY = "Xe"
_BRANCHING_DUMMY = "Kr"


def _build_mol_and_graph(mof, indices, ignore_hidden_indices=True, add_additional_site=True):
    indices = set(indices)
    sites_and_indices = _sites_and_classified_indices_from_indices(mof, indices)
    if ignore_hidden_indices:
        relevant_indices = indices - set(sites_and_indices.hidden_vertices)
    else:
        relevant_indices = indices

    selected_positions, selected_species, selected_edge_indices = [], [], []

    # give us a quick and easy way to filter the edges we want to keep
    # for this reason we will use both edge "partners" as keys
    # and have a list of indices in the original edges list as the value
    edge_dict = defaultdict(list)
    for i, egde in enumerate(sites_and_indices.edges):
        edge_partner_1, edge_partner_2 = egde
        edge_dict[edge_partner_1].append(i)
        edge_dict[edge_partner_2].append(i)

    for i, index in enumerate(indices):
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

    mol, mapping = wrap_molecule(
        list(relevant_indices), mof, add_additional_site=add_additional_site
    )

    return mol, molecule_graph, mapping


def _extract_and_wrap(node_indices, all_branching_indices, all_binding_indices, mof):
    # first, create a molecule and molecule graph from those indices
    # that intersect with the node indices
    branching_in_node = [i for i in all_branching_indices if i in node_indices]
    binding_in_node = [i for i in all_binding_indices if i in node_indices]

    mol, graph, mapping = _build_mol_and_graph(
        mof, list(node_indices) + branching_in_node + binding_in_node
    )

    # now, also create one molecule from the node indices, where we replace
    # binding and branching indices with dummy atoms
    node_metal_atoms = [i for i in node_indices if i in mof.metal_indices]
    binding_to_node_metal = set(sum([mof.get_neighbor_indices(i) for i in node_metal_atoms], []))
    binding_to_node_metal = binding_to_node_metal.intersection(all_binding_indices)

    branching_to_node_metal = set(
        sum(
            [mof.get_neighbor_indices(i) for i in node_metal_atoms if i in all_branching_indices],
            [],
        )
    )
    binding_neighbors = sum([mof.get_neighbor_indices(i) for i in binding_to_node_metal], [])
    branching_to_binding = set(binding_neighbors).intersection(all_branching_indices)

    # Now, make a copy of the structure and replace the indices with dummy atoms
    dummy_structure = Structure.from_sites(mof.structure.sites)
    for i in binding_to_node_metal:
        dummy_structure.replace(i, _BINDING_DUMMY)
    for i in branching_to_binding | branching_to_node_metal:
        dummy_structure.replace(i, _BRANCHING_DUMMY)
    dummy_branching_sites = branching_to_binding | branching_to_node_metal
    mol_w_dummy, graph_w_dummy, mapping_w_dummy = _build_mol_and_graph(
        mof,
        list(node_indices)
        + list(binding_to_node_metal)
        + list(branching_to_binding)
        + list(branching_to_node_metal),
        ignore_hidden_indices=False,
        add_additional_site=False,
    )

    inverse_mapping = {v[0]: k for k, v in mapping_w_dummy.items()}
    # let's replace here now the atoms with the dummys as doing it beforehand might cause issues
    # (e.g. we do not have the distances for a cutoffdict)

    for i in branching_to_binding | branching_to_node_metal:
        if i not in node_indices & set(mof.metal_indices):
            mol_w_dummy._sites[inverse_mapping[i]] = Site(
                _BRANCHING_DUMMY,
                mol_w_dummy._sites[inverse_mapping[i]].coords,
                properties={"original_species": str(mol_w_dummy._sites[inverse_mapping[i]].specie)},
            )

    for i in binding_to_node_metal:
        mol_w_dummy._sites[inverse_mapping[i]] = Site(
            _BINDING_DUMMY,
            mol_w_dummy._sites[inverse_mapping[i]].coords,
            properties={"original_species": str(mol_w_dummy._sites[inverse_mapping[i]].specie)},
        )

    graph_w_dummy.molecule = mol_w_dummy

    return mol, graph, mapping, mol_w_dummy, graph_w_dummy, mapping_w_dummy, dummy_branching_sites


def node_from_mof_and_indices(cls, mof, node_indices, all_branching_indices, all_binding_indices):
    """Create a node from a MOF and a list of indices of different types."""
    node_indices = node_indices
    branching_indices = node_indices & all_branching_indices
    binding_indices = node_indices & all_binding_indices
    graph_branching_indices = branching_indices & node_indices

    (
        mol,
        graph,
        mapping,
        mol_w_dummy,
        graph_w_dummy,
        mapping_w_dummy,
        dummy_branching_sites,
    ) = _extract_and_wrap(
        node_indices=node_indices,
        all_branching_indices=all_branching_indices,
        all_binding_indices=all_binding_indices,
        mof=mof,
    )

    node = cls(
        molecule=mol,
        molecule_graph=graph,
        graph_branching_indices=graph_branching_indices,
        binding_indices=binding_indices,
        molecule_original_indices_mapping=mapping,
        dummy_molecule=mol_w_dummy,
        dummy_molecule_graph=graph_w_dummy,
        dummy_molecule_indices_mapping=mapping_w_dummy,
        dummy_branching_indices=dummy_branching_sites,
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
    ):
        """Build a node object from a MOF and some intermediate outputs of the fragmentation.

        Args:
            mof (MOF): The MOF to build the node from.
            node_indices (Set[int]): The indices of the nodes in the MOF.
            branching_indices (Set[int]): The indices of the branching points in the MOF
                that belong to this node.
            binding_indices (Set[int]): The indices of the binding points in the MOF
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
        )

    def __repr__(self) -> str:
        """Return a string representation of the node."""
        return f"Node ({self.composition})"
