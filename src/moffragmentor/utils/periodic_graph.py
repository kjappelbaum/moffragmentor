# -*- coding: utf-8 -*-
"""Methods on structure graphs"""


from pymatgen.core import Structure

from . import _not_relevant_structure_indices


def is_periodic(mof, indices):
    graph_ = mof.structure_graph.__copy__()
    graph_.structure = Structure.from_sites(graph_.structure.sites)
    to_delete = _not_relevant_structure_indices(mof.structure, indices)
    graph_.remove_nodes(to_delete)
    mols = graph_.get_subgraphs_as_molecules()
    return len(mols) > 0
