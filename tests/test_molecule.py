# -*- coding: utf-8 -*-
from moffragmentor.molecule import NonSbuMolecule


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
    indices = [194, 210, 538]
    mol = NonSbuMolecule.from_structure_graph_and_indices(graph, indices)
    assert str(mol) == "H2 O1"
