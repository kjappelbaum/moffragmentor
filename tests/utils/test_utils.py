import pytest

from moffragmentor.utils import get_molecule_mass


def test_get_molecule_mass(get_methane_molecule_and_graph):
    mol, graph = get_methane_molecule_and_graph

    assert get_molecule_mass(mol) == pytest.approx(16.04303, 0.1)
