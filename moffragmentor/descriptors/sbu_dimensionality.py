# -*- coding: utf-8 -*-
"""One important piece of information about the SBUs is their dimenaionality,
e.g., if they are clusters, if they form rods, or if they form sheets.
We use the algorithm proposed by Larsen et al.,
see https://journals.aps.org/prmaterials/pdf/10.1103/PhysRevMaterials.3.034003
"""
from copy import deepcopy

from pymatgen.analysis.dimensionality import get_dimensionality_larsen
from pymatgen.analysis.graphs import StructureGraph

from ..utils import remove_all_nodes_not_in_indices


def get_structure_graph_dimensionality(structure_graph: StructureGraph) -> int:
    """Use Larsen's algorithm to compute the dimensionality"""
    return get_dimensionality_larsen(structure_graph)


def get_sbu_dimensionality(mof, indices) -> int:
    """Computer the dimensionality of an SBU,
    characterized as a subset of indices of a MOF"""
    structure_graph = deepcopy(mof.structure_graph)
    remove_all_nodes_not_in_indices(structure_graph, indices)
    return get_structure_graph_dimensionality(structure_graph)
