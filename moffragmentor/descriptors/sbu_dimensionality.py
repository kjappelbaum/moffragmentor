# -*- coding: utf-8 -*-
"""One important piece of information about the SBUs is their dimenaionality,
e.g., if they are clusters, if they form rods, or if they form sheets.
We use the algorithm proposed by Larsen et al.,
see https://journals.aps.org/prmaterials/pdf/10.1103/PhysRevMaterials.3.034003
"""
from pymatgen.analysis.dimensionality import get_dimensionality_larsen
from pymatgen.analysis.graphs import StructureGraph


def get_structure_graph_dimensionality(structure_graph: StructureGraph) -> int:
    return get_dimensionality_larsen(structure_graph)
