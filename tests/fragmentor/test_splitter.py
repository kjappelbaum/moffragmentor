# -*- coding: utf-8 -*-
"""Testing the splitting module of the fragmentor subpackage"""
from collections import Counter

from structuregraph_helpers.subgraph import get_subgraphs_as_molecules

from moffragmentor.fragmentor.solventlocator import get_floating_solvent_molecules


def test_unbound_solvent_identification(get_p_linker_with_floating):
    """Test that we can correctly flag the floating solvents"""
    unique_mols, unique_graphs, unique_indices, _, _ = get_subgraphs_as_molecules(
        get_p_linker_with_floating.structure_graph
    )
    # we see pyrrolidinium, h2o and h3o+ as solvent,
    # see https://www.ccdc.cam.ac.uk/structures/Search?Ccdcid=MAGBON&DatabaseToSearch=Published
    assert len(unique_mols) == len(unique_graphs) == len(unique_indices) == 3
    compositions = [str(unique_mol.composition.alphabetical_formula) for unique_mol in unique_mols]

    assert "H2 O1" in compositions
    assert "C4 H10 N1" in compositions
    assert "O1" in compositions

    # Now, we'll return all in the unit cell
    mols, _, _, _, _ = get_subgraphs_as_molecules(
        get_p_linker_with_floating.structure_graph, return_unique=False
    )
    assert len(mols) == 24
    compositions = [str(mol.composition.alphabetical_formula) for mol in mols]
    composition_counter = Counter(compositions)
    assert composition_counter["C4 H10 N1"] == 4
    assert composition_counter["H2 O1"] == 8 * 2


def test_unbound_solvent_identification_li_mof(get_li_mof_with_floating):
    """Test flagging of unbound solvent in a MOF
    with a Li containing node"""
    unique_mols, unique_graphs, unique_indices, _, _ = get_subgraphs_as_molecules(
        get_li_mof_with_floating.structure_graph
    )
    assert len(unique_mols) == len(unique_graphs) == len(unique_indices) == 1
    compositions = [str(unique_mol.composition.alphabetical_formula) for unique_mol in unique_mols]
    assert "H2 O1" in compositions

    # Now, we'll return all in the unit cell
    mols, graphs, indices, _, _ = get_subgraphs_as_molecules(
        get_li_mof_with_floating.structure_graph, return_unique=False
    )

    assert len(mols) == len(graphs) == len(indices) == 8


def unbound_solvent_identification_li_mof(get_li_mof_with_floating):
    """Test if we can get a NonSbuMoleculeCollection from the MOF"""
    molecules = get_floating_solvent_molecules(get_li_mof_with_floating)
    assert len(molecules) == 8
    assert molecules.composition == {"H2 O1": 8}


def unbound_solvent_identification_acetate_zr_mof(get_acetate_zr_mof):
    molecules = get_floating_solvent_molecules(get_acetate_zr_mof)
    assert len(molecules) == 6
    assert molecules.composition == {"C2 H4 O2": 6}
