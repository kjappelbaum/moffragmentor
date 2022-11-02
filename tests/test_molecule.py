# -*- coding: utf-8 -*-
"""Testing functions related to molecule creation"""
from moffragmentor.molecule import NonSbuMolecule, NonSbuMoleculeCollection


def test_molecule_creation(get_methane_molecule_and_graph):
    """Make sure we can create a molecule object without any issues"""
    methane, methane_graph = get_methane_molecule_and_graph
    mol = NonSbuMolecule(methane, methane_graph, [0, 1, 2, 3])
    assert str(mol) == "C1 H4"
    assert len(mol) == 5


def test_molecule_creation_from_structure_graph(get_p_linker_with_floating):
    """Make sure we can create a molecule object from a structure graph"""
    mof = get_p_linker_with_floating
    graph = mof.structure_graph
    indices = [167, 159, 251]
    mol = NonSbuMolecule.from_structure_graph_and_indices(graph, indices)
    assert str(mol) == "H2 O1"


def test_get_molecule_collection(get_methane_molecule_and_graph):
    """Make sure we can create a collection of molecules"""
    methane, methane_graph = get_methane_molecule_and_graph
    mol = NonSbuMolecule(methane, methane_graph, [0, 1, 2, 3])
    molecule_collection = NonSbuMoleculeCollection([mol, mol])
    assert molecule_collection.composition == {"C1 H4": 2}

    # we should be able to sum the collections
    new_collection = molecule_collection + molecule_collection
    assert len(new_collection) == 4
    assert new_collection.composition == {"C1 H4": 4}

    # the old one is unmodified
    assert len(molecule_collection) == 2
