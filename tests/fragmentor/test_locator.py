# -*- coding: utf-8 -*-
"""Test locator modules"""
from collections import Counter

from pymatgen.core import Structure

from moffragmentor import MOF
from moffragmentor.fragmentor.linkerlocator import _create_linkers_from_node_location_result
from moffragmentor.fragmentor.nodelocator import create_node_collection, find_node_clusters
from moffragmentor.fragmentor.solventlocator import (
    _get_solvent_molecules_bound_to_node,
    get_all_bound_solvent_molecules,
)
from moffragmentor.molecule import NonSbuMoleculeCollection


def test_find_cu_i_ii_btc_clusters(get_cuiiibtc_mof):
    """Make sure that we do find pytest he two correct node types in Cu(I/II)-BTC"""
    mof = get_cuiiibtc_mof
    node_location_result = find_node_clusters(mof)
    for node in node_location_result.nodes:
        # Every node has four branching indices from the carboxy
        assert len(node & node_location_result.branching_indices) == 4
    assert len(node_location_result) == 5
    assert len(node_location_result.nodes) == 6  # fou r paddlewheels and two macrocylus
    nodes_sizes = [len(node) for node in node_location_result.nodes]
    assert len(set(nodes_sizes)) == 2
    size_counters = Counter(nodes_sizes)
    assert sorted(list(size_counters.values())) == [2, 4]
    assert sorted(list(size_counters.keys())) == [16, 20]


def test_find_porphyrin_mof_clusters(get_porphryin_mof):
    """Looking for the porphyrin MOF clusters"""
    mof = get_porphryin_mof
    node_location_result = find_node_clusters(mof)
    fragments = mof.fragment()
    assert len(node_location_result) == 5
    assert len(node_location_result.nodes) == 4
    assert len(fragments.nodes) == 4  # we by default break the rod nodes
    assert len(node_location_result.branching_indices) == 8 * 2
    assert fragments.linkers[0].search_pubchem()[0][0].cid == 58107362


def test_find_hypothetical_mof_clusters(get_hypothetical_mof):
    """Check if we can find the correct nodes in hypothetical MOFs"""
    mof = get_hypothetical_mof
    node_location_result = find_node_clusters(mof)
    assert len(node_location_result) == 5
    assert len(node_location_result.nodes) == 16
    node_lengths = [len(node) for node in node_location_result.nodes]
    assert len(set(node_lengths)) == 1
    assert node_lengths[0] == 26


def test_find_p_linker_floating_mof_clusters(get_p_linker_with_floating):
    """Check if we can deal with non-carboxy ligand and floating solvent"""
    mof = get_p_linker_with_floating
    node_location_result = find_node_clusters(mof)
    assert len(node_location_result) == 5
    assert len(node_location_result.nodes) == 4
    node_lengths = [len(node) for node in node_location_result.nodes]
    assert len(set(node_lengths)) == 1
    assert node_lengths[0] == 23


def test_find_li_mof_floating_mof_cluster(get_li_mof_with_floating):
    """A somewhat more complicated node,
    see https://pubs.rsc.org/en/content/articlelanding/2014/DT/c3dt53415d#!divAbstract"""
    mof = get_li_mof_with_floating
    node_location_result = find_node_clusters(mof)
    assert len(node_location_result) == 5
    assert len(node_location_result.nodes) == 2
    node_lengths = [len(node) for node in node_location_result.nodes]
    assert len(set(node_lengths)) == 1
    assert node_lengths[0] == 34


def test_find_rod_node_floating_mof_cluster(get_1d_node_with_floating):
    mof = get_1d_node_with_floating
    node_location_result = find_node_clusters(mof)
    assert len(node_location_result) == 5
    assert len(node_location_result.nodes) == 4
    node_lengths = [len(node) for node in node_location_result.nodes]
    assert len(set(node_lengths)) == 1
    assert node_lengths[0] == 20


def test_formate_mof(get_formate_structure_and_graph):
    s, sg = get_formate_structure_and_graph
    mof = MOF(s, sg)
    bbs = mof.fragment()
    assert len(bbs.nodes) == 2
    assert (
        len(bbs.capping_molecules) == 0
    )  # since there is no linker, we assign the formate as capping molecule
    assert bbs.linkers.composition == {"C1 H1 O2": 6}


def test__get_solvent_molecules_bound_to_node(get_li_mof_with_floating):
    mof = get_li_mof_with_floating
    node_indices = [
        0,
        128,
        130,
        2,
        132,
        4,
        134,
        6,
        136,
        138,
        140,
        142,
        144,
        146,
        148,
        150,
        152,
        154,
        156,
        158,
        164,
        166,
        48,
        50,
        52,
        54,
        68,
        70,
        92,
        94,
        104,
        106,
        116,
        118,
    ]
    solvent_molecules = _get_solvent_molecules_bound_to_node(mof, node_indices)
    assert isinstance(solvent_molecules, NonSbuMoleculeCollection)
    assert solvent_molecules.composition == {"H2 O1": 2}


def test_find_all_bound_solvent_molecules(get_li_mof_with_floating):
    mof = get_li_mof_with_floating
    node_collection = [
        {
            0,
            128,
            130,
            2,
            132,
            4,
            134,
            6,
            136,
            138,
            140,
            142,
            144,
            146,
            148,
            150,
            152,
            154,
            156,
            158,
            164,
            166,
            48,
            50,
            52,
            54,
            68,
            70,
            92,
            94,
            104,
            106,
            116,
            118,
        },
        {
            1,
            129,
            3,
            131,
            133,
            5,
            135,
            7,
            137,
            139,
            141,
            143,
            145,
            147,
            149,
            151,
            153,
            155,
            157,
            159,
            165,
            167,
            49,
            51,
            53,
            55,
            69,
            71,
            93,
            95,
            105,
            107,
            117,
            119,
        },
    ]
    solvent_molecules = get_all_bound_solvent_molecules(mof, node_collection)
    assert isinstance(solvent_molecules, NonSbuMoleculeCollection)
    assert solvent_molecules.composition == {"H2 O1": 4}


def test_find_node_cluster_acetate_zr_mof(get_acetate_zr_mof):
    """Nice (Zr6)2 node,
    see https://pubs.rsc.org/en/content/articlelanding/2018/CC/C8CC00507A#!divAbstract"""
    mof = get_acetate_zr_mof
    node_location_result = find_node_clusters(mof)
    assert len(node_location_result) == 5
    assert len(node_location_result.nodes) == 1
    # there are only three carboxy that actually come from a linker in Zr6 part
    assert len(node_location_result.branching_indices) == 24
    node_lengths = [len(node) for node in node_location_result.nodes]
    assert len(set(node_lengths)) == 1


def test_find_all_bound_solvent_molecules_acetate_zr_mof(get_acetate_zr_mof):
    mof = get_acetate_zr_mof
    node_collection = [
        {
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
            118,
            119,
            122,
            123,
            126,
            127,
            130,
            131,
            134,
            135,
            138,
            139,
            142,
            143,
            144,
            145,
            170,
            171,
            184,
            185,
            188,
            189,
            192,
            193,
            202,
            203,
            204,
            205,
            206,
            207,
            208,
            209,
            210,
            211,
            212,
            213,
            214,
            215,
            216,
            217,
            218,
            219,
            220,
            221,
            222,
            223,
            224,
            225,
            226,
            227,
            228,
            229,
            230,
            231,
            232,
            233,
            234,
            235,
            236,
            237,
            238,
            239,
            240,
            241,
            242,
            243,
            244,
            245,
            246,
            247,
            248,
            249,
            250,
            251,
            252,
            253,
            254,
            255,
            256,
            257,
            258,
            259,
            260,
            261,
            262,
            263,
            264,
            265,
        }
    ]
    solvent_molecules = get_all_bound_solvent_molecules(mof, node_collection)
    assert isinstance(solvent_molecules, NonSbuMoleculeCollection)
    assert solvent_molecules.composition == {"C2 H3 O2": 8}


def test__create_linkers_from_node_location_result(get_hkust_mof):
    mof = get_hkust_mof
    unbound_solvent = NonSbuMoleculeCollection([])
    node_location_result = find_node_clusters(mof)
    node_collection = create_node_collection(mof, node_location_result)
    linkers = _create_linkers_from_node_location_result(
        mof, node_location_result, node_collection, unbound_solvent, unbound_solvent
    )
    assert len(linkers) == 8
    linker_lengths = [len(linker) for linker in linkers]
    assert len(set(linker_lengths)) == 1


def test__get_node_cluster_across_pbc(get_across_periodic_boundary_node):
    mof = get_across_periodic_boundary_node
    node_location_result = find_node_clusters(mof)
    assert len(node_location_result.nodes) == 2  # we have 8 Zn in unit cell and Zn4 nodes
    sites = []

    for site in node_location_result.nodes[0]:
        sites.append(mof.structure[site])
    s = Structure.from_sites(sites)
    assert str(s.composition) == "Zn4 P2 H6 C6 O18"
