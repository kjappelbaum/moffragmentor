# -*- coding: utf-8 -*-
"""Testing the splitting module of the fragmentor subpackage"""
from collections import Counter

from moffragmentor.fragmentor.splitter import get_subgraphs_as_molecules


def test_unbound_solvent_identification(get_p_linker_with_floating):
    """Test that we can correctly flag the floating solvents"""
    unique_mols, unique_graphs, unique_indices = get_subgraphs_as_molecules(
        get_p_linker_with_floating.structure_graph
    )
    # we see pyrrolidinium, h2o and h3o+ as solvent, see https://www.ccdc.cam.ac.uk/structures/Search?Ccdcid=MAGBON&DatabaseToSearch=Published
    assert len(unique_mols) == len(unique_graphs) == len(unique_indices) == 3
    compositions = [
        str(unique_mol.composition.alphabetical_formula) for unique_mol in unique_mols
    ]

    assert "H2 O1" in compositions
    assert "C4 H10 N1" in compositions
    assert "O1" in compositions

    # Now, we'll return all in the unit cell
    mols, graphs, indices = get_subgraphs_as_molecules(
        get_p_linker_with_floating.structure_graph, return_unique=False
    )
    assert len(mols) == 48
    compositions = [str(mol.composition.alphabetical_formula) for mol in mols]
    composition_counter = Counter(compositions)
    print(composition_counter)
    assert composition_counter["C4 H10 N1"] == 8
    assert composition_counter["H2 O1"] == 8 * 4
