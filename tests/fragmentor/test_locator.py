# -*- coding: utf-8 -*-
from collections import Counter

from moffragmentor.fragmentor.locator import (
    find_node_clusters,
    get_solvent_molecules_bound_to_node,
)
from moffragmentor.molecule import NonSbuMoleculeCollection


def test_find_cu_i_ii_btc_clusters(get_cuiiibtc_mof):
    """Make sure that we do find the two correct node types in Cu(I/II)-BTC"""
    mof = get_cuiiibtc_mof
    node_location_result = find_node_clusters(mof)
    assert len(node_location_result) == 3
    assert len(node_location_result.nodes) == 6  # four paddlewheels and two macrocylus
    nodes_sizes = [len(node) for node in node_location_result.nodes]
    assert len(set(nodes_sizes)) == 2
    size_counters = Counter(nodes_sizes)
    assert sorted(list(size_counters.values())) == [2, 4]
    assert sorted(list(size_counters.keys())) == [16, 20]


def test_find_porphyrin_mof_clusters(get_porphryin_mof):
    """Looking for the porphyrin MOF clusters, here we should only find two nodes in the unit cell"""
    mof = get_porphryin_mof
    node_location_result = find_node_clusters(mof)
    assert len(node_location_result) == 3
    assert len(node_location_result.nodes) == 2
    assert len(node_location_result.branching_indices) == 8 * 2


def test_find_hypothetical_mof_clusters(get_hypothetical_mof):
    """Check if we can find the correct nodes in hypothetical MOFs"""
    mof = get_hypothetical_mof
    node_location_result = find_node_clusters(mof)
    assert len(node_location_result) == 3
    assert len(node_location_result.nodes) == 16
    node_lengths = [len(node) for node in node_location_result.nodes]
    assert len(set(node_lengths)) == 1
    assert node_lengths[0] == 24


def test_find_p_linker_floating_mof_clusters(get_p_linker_with_floating):
    """Check if we can deal with non-carboxy ligand and floating solvent"""
    mof = get_p_linker_with_floating
    node_location_result = find_node_clusters(mof)
    assert len(node_location_result) == 3
    assert len(node_location_result.nodes) == 8
    node_lengths = [len(node) for node in node_location_result.nodes]
    assert len(set(node_lengths)) == 1
    assert (
        node_lengths[0] == 25
    )  # ToDo: think more carefully if we want to deal this way with the µ1-carboxy


def test_find_li_mof_floating_mof_cluster(get_li_mof_with_floating):
    """A somewhat more complicated node,
    see https://pubs.rsc.org/en/content/articlelanding/2014/DT/c3dt53415d#!divAbstract"""
    mof = get_li_mof_with_floating
    node_location_result = find_node_clusters(mof)
    print(node_location_result.nodes)
    assert len(node_location_result) == 3
    assert len(node_location_result.nodes) == 2
    node_lengths = [len(node) for node in node_location_result.nodes]
    assert len(set(node_lengths)) == 1
    assert node_lengths[0] == 34


def test_find_rod_node_floating_mof_cluster(get_1d_node_with_floating):
    mof = get_1d_node_with_floating
    node_location_result = find_node_clusters(mof)
    assert len(node_location_result) == 3
    assert len(node_location_result.nodes) == 4
    node_lengths = [len(node) for node in node_location_result.nodes]
    assert len(set(node_lengths)) == 1
    assert node_lengths[0] == 20


def test_get_solvent_molecules_bound_to_node(get_li_mof_with_floating):
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
    solvent_molecules = get_solvent_molecules_bound_to_node(mof, node_indices)
    assert isinstance(solvent_molecules, NonSbuMoleculeCollection)
    assert solvent_molecules.composition == {"H2 O1": 2}
