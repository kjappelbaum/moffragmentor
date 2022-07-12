# -*- coding: utf-8 -*-
"""Methods on structure graphs"""
from typing import List

import networkx as nx
from pymatgen.core import Structure

from . import _not_relevant_structure_indices


def is_periodic(mof, indices):
    graph_ = mof.structure_graph.__copy__()
    graph_.structure = Structure.from_sites(graph_.structure.sites)
    to_delete = _not_relevant_structure_indices(mof.structure, indices)
    graph_.remove_nodes(to_delete)
    mols = graph_.get_subgraphs_as_molecules()
    return len(mols) > 0


def _get_leaf_nodes(graph: nx.Graph) -> List[int]:
    return [node for node in graph.nodes() if graph.degree(node) == 1]


def _get_number_of_leaf_nodes(graph: nx.Graph) -> int:
    return len(_get_leaf_nodes(graph))
