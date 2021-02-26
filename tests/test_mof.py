# -*- coding: utf-8 -*-
import os

import pytest
from pymatgen import Structure

from moffragmentor import MOF

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def check_mof_creation_from_cif(get_cuiiibtc_mof):
    mof = get_cuiiibtc_mof
    assert isinstance(mof, object)
    assert isinstance(mof.structure, Structure)


@pytest.mark.julia
def test_topology_computation(get_hkust_mof):
    mof = get_hkust_mof
    assert mof.topology == "tbo"


def test_is_terminal(get_cuiiibtc_mof):
    """An atom is terminal when it only has one neighbor"""
    mof =get_cuiiibtc_mof
    assert mof._is_terminal(32)
    assert not mof._is_terminal(129)
    assert not mof._is_terminal(176)

def test_node_indices(get_cuiiibtc_mof):
    mof = get_cuiiibtc_mof
    mof.fragmentation_method = 'oxo'
    assert mof.fragmentation_method == 'oxo'
    assert mof._node_indices is None

    assert mof.node_indices == set(mof.metal_indices)
   
    mof.fragmentation_method = 'all_node'
    assert mof.node_indices == set(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            58,
            61,
            64,
            67,
            70,
            73,
            76,
            79,
            82,
            85,
            88,
            91,
            94,
            97,
            100,
            103,
            104,
            107,
            110,
            113,
            116,
            119,
            122,
            125,
            128,
            129,
            130,
            131,
            132,
            133,
            134,
            135,
            136,
            137,
            138,
            139,
            140,
            141,
            142,
            143,
            144,
            145,
            146,
            147,
            148,
            149,
            150,
            151,
            152,
            153,
            154,
            155,
            156,
            157,
            158,
            159,
            160,
            161,
            162,
            163,
            164,
            165,
            166,
            167,
            168,
            169,
            170,
            171,
            172,
            173,
            174,
            175,
        ]
    )

def test_linker_indices(get_cuiiibtc_mof):
    mof = get_cuiiibtc_mof
    mof.fragmentation_method = 'oxo'
    linker_indices = mof.linker_indices
    for linker_index in linker_indices:
        assert str(mof.structure[linker_index].specie) != 'Cu'
        assert linker_index not in mof.node_indices
        assert linker_index not in mof._solvent_indices

    mof = get_cuiiibtc_mof
    mof.fragmentation_method = 'all_node'
    linker_indices = mof.linker_indices
    for linker_index in linker_indices:
        assert str(mof.structure[linker_index].specie) != 'Cu'
        assert str(mof.structure[linker_index].specie) != 'O'
        assert linker_index not in mof.node_indices
        assert linker_index not in mof._solvent_indices


def test_label_structure(get_cuiiibtc_mof):
    mof = get_cuiiibtc_mof
    mof._get_node_indices()
    mof._label_structure()
    label_site_counter = 0
    sites = []
    for i in range(len(mof.structure)):
        if mof.structure[i].properties == {"binding": True}:
            label_site_counter += 1
            sites.append(i)
    assert label_site_counter == 2 * len(mof._connecting_node_indices)
    for index in sites:
        assert str(mof.structure[index].specie) == 'C'

    mof = get_cuiiibtc_mof
    mof.fragmentation_method = 'oxo'
    assert  mof.fragmentation_method == 'oxo'
    mof._get_node_indices()
    mof._label_structure()
    assert len(mof._connecting_node_indices) == len(mof.metal_indices)
    assert len(mof.node_indices) +  len(mof.linker_indices) + len(sum(mof._solvent_indices, [])) == len(mof.structure)

    sites = []
    # one important factor for labeling is that we get also the neighbors of the connecting 
    # atoms right
    for index in mof._connecting_node_indices:
        for neighbor in mof.get_neighbor_indices(index):
            assert str(mof.structure[neighbor].specie) == 'O' 

    # i think i currently have the issue due to the Os in the Cu(I) ring 
    for i in range(len(mof.structure)):
        if mof.structure[i].properties == {"binding": True}:
            sites.append(i)
    # assert len(sites) == 112

    for index in sites:
        element = str(mof.structure[index].specie)
        assert element == 'O' or element == 'Cu'
        assert index not in set(sum(mof._solvent_indices, []))


def test_fragmentation(get_hkust_mof, get_cuiiibtc_mof):
    mof = get_hkust_mof
    linkers, nodes = mof.fragment()
    assert len(linkers) == 1
    assert len(nodes) == 1
    assert linkers[0].smiles == "[C]1=C[C]=C[C]=C1"

    mof2 = get_cuiiibtc_mof
    linkers, nodes = mof2.fragment()
    assert len(linkers) == 1

    assert len(nodes) == 2
    assert len(nodes[0].molecule) == 16
    assert len(nodes[1].molecule) == 14
    assert linkers[0].smiles == "[C]1=C[C]=C[C]=C1"
