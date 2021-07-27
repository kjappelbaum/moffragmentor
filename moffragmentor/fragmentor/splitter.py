# -*- coding: utf-8 -*-
"""This module focusses on the extraction of pymatgen Molecules from a structure for which we know the branching points / node/linker indices"""
from collections import namedtuple
from typing import List, Tuple

import networkx as nx
import numpy as np
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.core import Molecule

from ..utils import (
    _get_cartesian_coords,
    _get_molecule_edge_label,
    _get_vertices_of_smaller_component_upon_edge_break,
    _reindex_list_of_tuple,
    revert_dict,
)
from .solventlocator import _locate_bound_solvent

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

    solvent = _locate_bound_solvent(mof, indices)
    any_bound_solvent_index = set(sum(solvent["solvent_indices"], []))
    # Here we now need to get the bound solvent components to make sure we can exlude them
    # ToDo: we can then export this result and skip the calculation in the fragment function

    for bridge in new_bridges:
        component = _get_vertices_of_smaller_component_upon_edge_break(
            graph.graph.to_undirected(), bridge
        )
        indices = [new_index_to_old_index[i] for i in component]
        if not set(indices) & any_bound_solvent_index:
            persistent_non_metal_bridged_components.append(indices)

    persistent_non_metal_bridged_components_old_idx = []
    for subset in persistent_non_metal_bridged_components:
        persistent_non_metal_bridged_components_old_idx.append(subset)

    return _SitesAndIndicesOutput(
        np.array(new_positions),
        species,
        edges,
        index_mapping,
        hidden_vertices,
        persistent_non_metal_bridged_components_old_idx,
    )
