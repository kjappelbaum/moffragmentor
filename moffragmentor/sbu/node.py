# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import Set

import numpy as np
from pymatgen.analysis.graphs import MoleculeGraph

from ..utils import (
    _not_relevant_structure_indices,
    connected_mol_from_indices,
    get_edge_dict,
)
from .sbu import SBU


class Node(SBU):
    @classmethod
    def from_mof_and_indices(
        cls,
        mof,
        node_indices: Set[int],
        branching_indices: Set[int],
        binding_indices: Set[int],
    ):
        graph_ = deepcopy(mof.structure_graph)
        to_delete = _not_relevant_structure_indices(mof.structure, node_indices)
        graph_.remove_nodes(to_delete)

        mol = connected_mol_from_indices(mof, node_indices)
        graph = MoleculeGraph.with_edges(mol, get_edge_dict(graph_))

        center = np.mean(mol.cart_coords, axis=0)

        node = cls(
            mol,
            graph,
            center,
            branching_indices & node_indices,
            binding_indices & node_indices,
            node_indices,
        )
        return node
