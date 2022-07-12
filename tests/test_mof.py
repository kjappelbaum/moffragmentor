# -*- coding: utf-8 -*-
"""Testing some attributes of the MOF class"""
import os

from pymatgen.core import Structure

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def check_mof_creation_from_cif(get_cuiiibtc_mof):
    mof = get_cuiiibtc_mof
    assert isinstance(mof, object)
    assert isinstance(mof.structure, Structure)


def test_is_terminal(get_cuiiibtc_mof):
    """An atom is terminal when it only has one neighbor"""
    mof = get_cuiiibtc_mof
    assert mof._is_terminal(32)  # pylint:disable=protected-access
    assert not mof._is_terminal(129)  # pylint:disable=protected-access
    assert not mof._is_terminal(176)  # pylint:disable=protected-access


def test__leads_to_terminal(get_cuiiibtc_mof):
    mof = get_cuiiibtc_mof
    assert mof._leads_to_terminal((15, 183))  # pylint:disable=protected-access


def test_fragmentation(get_cuiiibtc_mof):
    mof = get_cuiiibtc_mof
    fragments = mof.fragment()
    # topocryst.com fails here
    assert fragments.net_embedding.rcsr_code == "mmm"
    assert fragments.net_embedding.space_group == "P4/nmm"


def test_fragmentation_single_metal(get_single_metal_mof):
    mof = get_single_metal_mof
    fragments = mof.fragment()
    assert len(fragments.nodes) == 4
    assert fragments.linkers[0].search_pubchem()[0][0].cid == 60197031
    assert fragments.net_embedding.rcsr_code == "dia"


def test_fragmentation_ag_n_compound(get_agn_mof):
    mof = get_agn_mof
    fragments = mof.fragment()
    assert fragments.net_embedding.rcsr_code == "bex"
