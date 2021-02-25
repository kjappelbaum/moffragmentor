# -*- coding: utf-8 -*-
import os

import pytest
from pymatgen import Structure

from moffragmentor import MOF

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def check_mof_creation_from_cif():
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "HKUST-1.cif"))
    assert isinstance(mof, object)
    assert isinstance(mof.structure, Structure)


@pytest.mark.julia
def test_topology_computation():
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "HKUST-1.cif"))
    assert mof.topology == "tbo"


def test_is_terminal():
    """An atom is terminal when it only has one neighbor"""
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "KAJZIH_freeONLY.cif"))
    assert mof._is_terminal(32)
    assert not mof._is_terminal(129)
    assert not mof._is_terminal(176)


# def test_fragmentation():
#     mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "HKUST-1.cif"))
#     linkers, nodes = mof.fragment()
#     assert len(linkers) == 1
#     assert len(nodes) == 1
#     assert linkers[0].smiles == "[O]C(=O)c1cc(cc(c1)C(=O)[O])C(=O)[O]"

#     mof2 = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "KAJZIH_freeONLY.cif"))
#     linkers, nodes = mof2.fragment()
#     assert len(linkers) == 1
#     assert len(nodes) == 2
#     assert len(nodes[0].molecule) == 4
#     assert len(nodes[1].molecule) == 8
#     assert linkers[0].smiles == "[O]C(=O)c1cc(cc(c1)C(=O)[O])C(=O)[O]"
