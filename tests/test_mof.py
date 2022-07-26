# -*- coding: utf-8 -*-
"""Testing some attributes of the MOF class"""
import os

import pytest
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


@pytest.mark.slow
def test_fragmentation(get_cuiiibtc_mof):
    mof = get_cuiiibtc_mof
    fragments = mof.fragment()
    # topocryst.com fails here
    assert fragments.net_embedding.rcsr_code == "mmm"


@pytest.mark.slow
def test_fragmentation_single_metal(get_single_metal_mof):
    mof = get_single_metal_mof
    fragments = mof.fragment()
    assert len(fragments.nodes) == 4
    assert fragments.linkers[0].is_edge
    # assert (
    #     fragments.linkers[1].search_pubchem()[0][0].cid
    #     == fragments.linkers[0].search_pubchem()[0][0].cid
    #     == 60197031
    # )
    # this is actually quite an interesting one as
    # one carboxy is not coordinated.
    assert fragments.net_embedding.rcsr_code == "dia"


@pytest.mark.slow
def test_fragmentation_ag_n_compound(get_agn_mof):
    mof = get_agn_mof
    fragments = mof.fragment()
    pubchem_res = fragments.linkers[0].search_pubchem()
    assert pubchem_res[0][0].cid == 59188427

    # actually not sure if this test was correct
    # topos also just gives "unknown topology" for all clustering techniques
    # as does mofid
    # in particular, bex is also a 2D periodic compound
    # assert fragments.net_embedding.rcsr_code == "bex"

def _get_distances(cs):
    distances = []
    for i, site in enumerate(cs):
        for j, site2 in enumerate(cs):
            if i != j:
                distances.append(site.distance(site2))
    return distances
    
@pytest.mark.slow
def test_hkust(get_hkust_mof):
    mof = get_hkust_mof
    fragments = mof.fragment()
    assert fragments.net_embedding.rcsr_code == "tbo"
    for linker in fragments.linkers:
        cs = linker._get_connected_sites_structure()
        distances = _get_distances(cs)
        assert max(distances) < 8
    
    for node in fragments.nodes:
        cs = node._get_connected_sites_structure()
        distances = _get_distances(cs)
        assert max(distances) < 8


@pytest.mark.slow
def test_mof5(get_mof5):
    mof = get_mof5
    fragments = mof.fragment()
    assert fragments.net_embedding.rcsr_code == "pcu"
